#include "thread_pool.h"
#include "postprocess.h"
#include "preprocess.h"
#include "yolo_infer.h"
#include "thread_pool.h"
#include "mem_ctrl.h"
#include "config.h"

#define GPU_BLOCK_THREADS 512


//该情况假设batchsize=1,  (116 = 4 + 80 + 32)
const int det_head_dim[3] = {1, 8400, 116};       //检测头的维度
const int seg_head_dim[4] = {1, 32, 160, 160};    //分割头的维度

const int NUM_BOX_ELEMENT = 8;  // left, top, right, bottom, confidence, class, keepflag, row_index(output)
const int MAX_IMAGE_BOXES = 1024;

// extern memory_ctrl::Memory<uint8_t> preprocess_buffer; //预处理相关的内存申请 (affine matrix)
// extern AffineMatrix affine_matrixs; 

static dim3 grid_dims(int numJobs) {
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}


static dim3 block_dims(int numJobs) {
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static __host__ __device__ void affine_project(float *matrix, 
                                               float x, float y, 
                                               float *ox, float *oy) 
{
  *ox = matrix[0] * x + matrix[1] * y + matrix[2];
  *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}


static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom) {
  float cleft = max(aleft, bleft);
  float ctop = max(atop, btop);
  float cright = min(aright, bright);
  float cbottom = min(abottom, bbottom);

  float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
  if (c_area == 0.0f) return 0.0f;

  float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
  float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
  return c_area / (a_area + b_area - c_area);
}

///////////////////////// GPU Part /////////////////////////

__global__ void decode_kernel_v8(float *predict, int num_bboxes, int num_classes,
                                  int output_cdim, float confidence_threshold,
                                  float *invert_affine_matrix, float *parray,
                                  int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float *class_confidence = pitem + 4;
  float confidence = *class_confidence++;
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width = *pitem++;
  float height = *pitem++;
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;  // 1 = keep, 0 = ignore
  *pout_item++ = position;
}


__global__ void decode_single_mask_kernel(int left, int top, float *mask_weights,
                                          float *mask_predict, int mask_width,
                                          int mask_height, uint8_t *mask_out,
                                          int mask_dim, int out_width, int out_height) {
  //mask_weights   (32)                                    
  //mask_predict   (32 * 160 * 160)

  // mask_predict to mask_out
  // mask_weights @ mask_predict
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= out_width || dy >= out_height) return;

  int sx = left + dx;
  int sy = top + dy;
  if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
    mask_out[dy * out_width + dx] = 0;
    return;
  }

  float cumprod = 0;
  for (int ic = 0; ic < mask_dim; ++ic) {
    float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
    float wval = mask_weights[ic];  //每个mask_predict的一个mask(160*160)的权重
    cumprod += cval * wval;
  }

  float alpha = 1.0f / (1.0f + exp(-cumprod));  //sigmoid估计mask概率
  mask_out[dy * out_width + dx] = alpha * 255;
}


__global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold) {
  int position = (blockDim.x * blockIdx.x + threadIdx.x);
  int count = min((int)*bboxes, MAX_IMAGE_BOXES);
  if (position >= count) return;

  // left, top, right, bottom, confidence, class, keepflag
  float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
  for (int i = 0; i < count; ++i) {
    float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
    if (i == position || pcurrent[5] != pitem[5]) continue;

    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && i < position) continue;

      float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                          pitem[2], pitem[3]);

      if (iou > threshold) {
        pcurrent[6] = 0;  // 1=keep, 0=ignore
        return;
      }
    }
  }
}


//将engine输出的rrrgggbbb转化为bgrbgrbgr格式
__global__ void rrrgggbbb_2_bgrbgrbgr_cuda(float *input, float *output, int width, int height) {
  // 获取线程的全局索引
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // 检查是否在图像范围内
  if (x < width && y < height) {

    int area = width * height;
    float *psrc_r = input + y * width + x;
    float *psrc_g = psrc_r + area;
    float *psrc_b = psrc_g + area;

    // 计算像素位置
    int idx = (y * width + x) * 3;

    // 将通道值乘以以255
    output[idx] = *psrc_b * 255.0f;
    output[idx + 1] = *psrc_g * 255.0f;
    output[idx + 2] = *psrc_r * 255.0f;
  }
}


/// @brief 
/// @param engine_out_gpu engine输出(rrrgggbbb)
/// @param output_w       engine输出width
/// @param output_h       engine输出height
/// @return               opencv输入(bgrbgrbgr)
cv::Mat rrrgggbbb_2_bgrbgrbgr(float* engine_out_gpu, 
                              const int output_w,
                              const int output_h,
                              const cv::Mat &img)
{

  //compute IM (640, 640)-->(img.cols, img.rows)
  float scale_x = output_w / (float)img.cols;
  float scale_y = output_h / (float)img.rows;
  float scale   = std::min(scale_x, scale_y);
  float ox      = -scale * img.cols * 0.5 + output_w * 0.5 + scale * 0.5 - 0.5;
  float oy      = -scale * img.rows * 0.5 + output_h * 0.5 + scale * 0.5 - 0.5;
  cv::Mat M     = (cv::Mat_<float>(2, 3)<<scale, 0, ox, 0, scale, oy);    
  
  cv::Mat IM, bgrbgrbgr_resized;
  cv::invertAffineTransform(M, IM); 

  //rrrgggbbb转化到bgrbgrbgr
  size_t engine_out_size = output_w * output_h * 3 * sizeof(float);

  float *cache_engine_out_gpu = nullptr;  //(rrrgggbbb)
  float *cache_bgrbgrbgr_gpu = nullptr;   //(bgrbgrbgr)
  float *cache_bgrbgrbgr_cpu = (float *)malloc(engine_out_size);   //(bgrbgrbgr)

  cudaStream_t stream_ = nullptr;  
  cudaStreamCreate(&stream_);
  CUDA_CHECK(cudaMalloc((void**)&cache_engine_out_gpu, engine_out_size));
  CUDA_CHECK(cudaMalloc((void**)&cache_bgrbgrbgr_gpu, engine_out_size));
  // GPU-->GPU转移内存
  CUDA_CHECK(cudaMemcpyAsync(cache_engine_out_gpu, engine_out_gpu, engine_out_size, 
                             cudaMemcpyDeviceToDevice, stream_));

  // 定义块大小和网格大小
  dim3 blockSize(32, 32);
  dim3 gridSize((output_w + blockSize.x - 1) / blockSize.x, (output_h + blockSize.y - 1) / blockSize.y);


  // 调用 CUDA 核函数
  rrrgggbbb_2_bgrbgrbgr_cuda<<<gridSize, blockSize, 0, stream_>>>(cache_engine_out_gpu, 
                                                                  cache_bgrbgrbgr_gpu, 
                                                                  output_w, 
                                                                  output_h);

  // 将结果从设备内存复制到主机内存
  CUDA_CHECK(cudaMemcpyAsync(cache_bgrbgrbgr_cpu, cache_bgrbgrbgr_gpu, engine_out_size, 
                            cudaMemcpyDeviceToHost, stream_));

  //CUDA流同步
  CUDA_CHECK(cudaStreamSynchronize(stream_)); 

  cv::Mat bgrbgrbgr_mat(output_h, output_w, CV_32FC3, cache_bgrbgrbgr_cpu);  
  bgrbgrbgr_mat.convertTo(bgrbgrbgr_mat, CV_8UC3); 

  //将bgrbgrbgr_mat还原到原图大小
  cv::warpAffine(bgrbgrbgr_mat, bgrbgrbgr_resized, IM, img.size(), cv::INTER_LINEAR);  

  // 释放内存
  CUDA_CHECK(cudaFree(cache_engine_out_gpu));
  CUDA_CHECK(cudaFree(cache_bgrbgrbgr_gpu));
  free(cache_bgrbgrbgr_cpu);

  return bgrbgrbgr_resized;

}


static void decode_kernel_invoker(float *predict, int num_bboxes, int output_cdim,int num_classes, 
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                                  cudaStream_t stream) {
  
  auto grid = grid_dims(num_bboxes);
  auto block = block_dims(num_bboxes);

  decode_kernel_v8<<<grid, block, 0, stream>>>(
                                predict, num_bboxes, num_classes, 
                                output_cdim, confidence_threshold, 
                                invert_affine_matrix, parray, MAX_IMAGE_BOXES);
  
  grid = grid_dims(MAX_IMAGE_BOXES);
  block = block_dims(MAX_IMAGE_BOXES);
  fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold);
}

static void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                               int mask_width, int mask_height, uint8_t *mask_out,
                               int mask_dim, int out_width, int out_height, cudaStream_t stream) {
  // mask_weights is mask_dim(32 element) gpu pointer
  dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
  dim3 block(32, 32);

  decode_single_mask_kernel<<<grid, block, 0, stream>>>(
                              left, top, mask_weights, mask_predict, 
                              mask_width, mask_height, mask_out, mask_dim, 
                              out_width, out_height);
}




yolo::BoxArray post_process(float *output_detect_buffer_,   //Network输出检测头
                            float *output_segmant_buffer_,  //Network输出分割头
                            memory_ctrl::Memory<uint8_t> &pre_buffer,
                            AffineMatrix &affine_matrixs,
                            const float kConfThresh, 
                            const float kNmsThresh,
                            const int kNumClass,
                            const int kInputW,
                            const int kInputH){
  
  memory_ctrl::Memory<float> Output_box_array;           //box结果相关的内存申请 
  memory_ctrl::Memory<float> Output_detect_buffer;       //检测头相关的内存申请 (8400 * 116)
  memory_ctrl::Memory<float> Output_segmant_buffer;      //分割头相关的内存申请 (32 * 160 * 160)
  std::vector<std::shared_ptr<memory_ctrl::Memory<uint8_t>>> box_segment_cache;

  size_t size_detect  = det_head_dim[1] * det_head_dim[2] * sizeof(float);
  size_t size_segmant = seg_head_dim[1] * seg_head_dim[2] * seg_head_dim[3] * sizeof(float);
  size_t size_bbox_array = (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT + 32) * sizeof(float);

  //申请内存
  float *detect_device = Output_detect_buffer.gpu(size_detect);      //GPU端的检测buff (8400 * 116)
  float *segmant_device = Output_segmant_buffer.gpu(size_segmant);   //GPU端的分割buff (32 * 160 * 160)
  float *box_array_device = Output_box_array.gpu(size_bbox_array);   //GPU端的box_array
  float *box_array_host = Output_box_array.cpu(size_bbox_array);     //CPU端的box_array

  float *affine_matrix_device = (float *)pre_buffer.gpu();

  cudaStream_t stream_ = nullptr;  
  cudaStreamCreate(&stream_);   //流处理加速
  // 内存初始化
  CUDA_CHECK(cudaMemsetAsync(box_array_device, 0, sizeof(int), stream_));

  // GPU-->GPU转移内存
  CUDA_CHECK(cudaMemcpyAsync(detect_device, output_detect_buffer_, size_detect, 
                                cudaMemcpyDeviceToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(segmant_device, output_segmant_buffer_, size_segmant,
                                cudaMemcpyDeviceToDevice, stream_));

  //进入核函数的封装
  decode_kernel_invoker(detect_device, det_head_dim[1], det_head_dim[2], 
                        kNumClass, kConfThresh, kNmsThresh, affine_matrix_device, 
                        box_array_device, MAX_IMAGE_BOXES, stream_);

  // GPU-->CPU转移内存
  //检测head解码和nms完成，将结果返回CPU
  CUDA_CHECK(cudaMemcpyAsync(Output_box_array.cpu(), Output_box_array.gpu(),
                             Output_box_array.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_)); //CUDA流同步

  //利用Mask和Mask权重，求解Mask
  yolo::BoxArray arrout;
  int imemory = 0;
  float *parray = Output_box_array.cpu();
  int count = min(MAX_IMAGE_BOXES, (int)*parray); //box数量
  yolo::BoxArray &output = arrout;
  output.reserve(count);
  for (int i = 0; i < count; ++i) {  //遍历每个box
    float *pbox = parray + 1 + i * NUM_BOX_ELEMENT;
    int label = pbox[5];
    int keepflag = pbox[6];
    if (keepflag == 1) {  //box是否保留，1=keep, 0=ignore
      yolo::Box result_object_box(pbox[0],   //left
                                  pbox[1],   //top
                                  pbox[2],   //right
                                  pbox[3],   //bottom
                                  pbox[4],   //confidence
                                  label); //构造函数来初始化box结果
      
      //分割部分
      int row_index = pbox[7];
      int mask_dim = seg_head_dim[1];
      float *mask_weights = detect_device + 
                            row_index * det_head_dim[2] + 
                            kNumClass + 4;

      float *mask_head_predict = Output_segmant_buffer.gpu();
      float left, top, right, bottom;
      float *i2d = affine_matrixs.i2d;
      //box的左上和右下两个点，image原图仿射到640*640
      affine_project(i2d, pbox[0], pbox[1], &left, &top);    
      affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

      float box_width = right - left;
      float box_height = bottom - top;

      float scale_to_predict_x = seg_head_dim[3] / (float)kInputW;
      float scale_to_predict_y = seg_head_dim[2] / (float)kInputH;

      //mask左上的坐标
      float mask_left = left * scale_to_predict_x;  
      float mask_top = top * scale_to_predict_y;
      //mask的长宽
      int mask_out_width = box_width * scale_to_predict_x + 0.5f;
      int mask_out_height = box_height * scale_to_predict_y + 0.5f;

      if (mask_out_width > 0 && mask_out_height > 0) {
        if (imemory >= (int)box_segment_cache.size()) {
          box_segment_cache.push_back(std::make_shared<memory_ctrl::Memory<uint8_t>>());
        }

        int bytes_of_mask_out = mask_out_width * mask_out_height;
        auto box_segment_output_memory = box_segment_cache[imemory];
        result_object_box.seg =
            std::make_shared<yolo::InstanceSegmentMap>(mask_out_width, mask_out_height);

        uint8_t *mask_out_device = box_segment_output_memory->gpu(bytes_of_mask_out);
        uint8_t *mask_out_host = result_object_box.seg->data;
        decode_single_mask(mask_left, 
                           mask_top, 
                           mask_weights,
                           mask_head_predict ,
                           seg_head_dim[3], 
                           seg_head_dim[2], 
                           mask_out_device,
                           mask_dim, 
                           mask_out_width, 
                           mask_out_height, 
                           stream_);

        //mask_out_host是160*160的mask掩码
        CUDA_CHECK(cudaMemcpyAsync(mask_out_host, mask_out_device,
                                    box_segment_output_memory->gpu_bytes(),
                                    cudaMemcpyDeviceToHost, stream_));
      }
      
      output.emplace_back(result_object_box);
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream_));

  //释放内存
  {
    pre_buffer.release();
    Output_box_array.release();           
    Output_detect_buffer.release();    
    Output_segmant_buffer.release();
    for(int i=0;i<box_segment_cache.size();i++)
        box_segment_cache[i]->release();
  }

  return arrout;
}



//////////////////////////////////// Draw Box and Segmant  ////////////////////////////////////



const char *cocolabels[] = {"person",        "bicycle",      "car",
                            "motorcycle",    "airplane",     "bus",
                            "train",         "truck",        "boat",
                            "traffic light", "fire hydrant", "stop sign",
                            "parking meter", "bench",        "bird",
                            "cat",           "dog",          "horse",
                            "sheep",         "cow",          "elephant",
                            "bear",          "zebra",        "giraffe",
                            "backpack",      "umbrella",     "handbag",
                            "tie",           "suitcase",     "frisbee",
                            "skis",          "snowboard",    "sports ball",
                            "kite",          "baseball bat", "baseball glove",
                            "skateboard",    "surfboard",    "tennis racket",
                            "bottle",        "wine glass",   "cup",
                            "fork",          "knife",        "spoon",
                            "bowl",          "banana",       "apple",
                            "sandwich",      "orange",       "broccoli",
                            "carrot",        "hot dog",      "pizza",
                            "donut",         "cake",         "chair",
                            "couch",         "potted plant", "bed",
                            "dining table",  "toilet",       "tv",
                            "laptop",        "mouse",        "remote",
                            "keyboard",      "cell phone",   "microwave",
                            "oven",          "toaster",      "sink",
                            "refrigerator",  "book",         "clock",
                            "vase",          "scissors",     "teddy bear",
                            "hair drier",    "toothbrush"};


__global__ void inverse_WarpAffine_Kernel(const uchar* input, uchar* output, 
                                          const float* M, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float inv_x = M[0] * x + M[1] * y + M[2];
        float inv_y = M[3] * x + M[4] * y + M[5];

        int x0 = static_cast<int>(inv_x);
        int y0 = static_cast<int>(inv_y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        if (x0 >= 0 && y0 >= 0 && x1 < width && y1 < height) {
            float dx = inv_x - x0;
            float dy = inv_y - y0;

            output[y * width + x] = static_cast<uchar>((1 - dx) * (1 - dy) * input[y0 * width + x0] +
                                                       dx * (1 - dy) * input[y0 * width + x1] +
                                                       (1 - dx) * dy * input[y1 * width + x0] +
                                                       dx * dy * input[y1 * width + x1]);
        }
    }
}


cv::Mat inverse_WarpAffine_CUDA(const cv::Mat& input, const float* M, cv::Size size) {
    int width = input.cols;
    int height = input.rows;

    const dim3 blockDim(32, 32);
    const dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                       (height + blockDim.y - 1) / blockDim.y);

    uchar* d_input;
    uchar* d_output;
    float* d_M;

    cv::Mat output(size, CV_8UC1);

    CUDA_CHECK(cudaMalloc(&d_input, input.total()));
    CUDA_CHECK(cudaMalloc(&d_output, output.total()));
    CUDA_CHECK(cudaMalloc(&d_M, 6 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input.data, input.total(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M, 6 * sizeof(float), cudaMemcpyHostToDevice));

    //逆向warpaffine
    inverse_WarpAffine_Kernel<<<gridDim, blockDim>>>(d_input, 
                                                     d_output, 
                                                     d_M, 
                                                     output.cols, 
                                                     output.rows
                                                     );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data, d_output, output.total(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_M));

    return output;
}


// 0：表示关闭线程池
// 1：表示打开线程池

#define use_threadpool 0

std::mutex mtx_postprocess;

static void draw_mask(const cv::Mat& image, 
                      yolo::Box& obj, 
                      cv::Scalar& color,
                      const int kInputW,
                      const int kInputH,
                      AffineMatrix &affine_matrixs)
{

#if use_threadpool //打开线程池
  std::lock_guard<std::mutex> lock(mtx_postprocess);
#endif

  /////////////////////////////////////////// Step1 ///////////////////////////////////////////
  float left, top;
  float *i2d = affine_matrixs.i2d;

  //投影到（640，640）
  affine_project(i2d, obj.left, obj.top, &left, &top);    

  float scale_x_box = seg_head_dim[3] / (float)kInputW;
  float scale_y_box = seg_head_dim[2] / (float)kInputH;

  float mask_left = left * (scale_x_box);
  float mask_top = top * (scale_y_box);

  // compute IM
  float scale_x = kInputW / (float)image.cols;
  float scale_y = kInputH / (float)image.rows;
  float scale   = std::min(scale_x, scale_y);
  float ox      = -scale * image.cols * 0.5 + kInputW * 0.5 + scale * 0.5 - 0.5;
  float oy      = -scale * image.rows * 0.5 + kInputH * 0.5 + scale * 0.5 - 0.5;
  cv::Mat M     = (cv::Mat_<float>(2, 3)<<scale, 0, ox, 0, scale, oy);    
  
  cv::Mat IM;
  cv::invertAffineTransform(M, IM); 

  /////////////////////////////////////////// Step2 ///////////////////////////////////////////
  cv::Mat mask_map = cv::Mat::zeros(cv::Size(seg_head_dim[2], 
                                              seg_head_dim[3]), 
                                              CV_8UC1); //(160,160)
  cv::Mat small_mask = cv::Mat(obj.seg->height, obj.seg->width, CV_8UC1, obj.seg->data);//分割的mask
  cv::Rect roi(mask_left, mask_top, obj.seg->width, obj.seg->height);
  small_mask.copyTo(mask_map(roi));
  cv::resize(mask_map, mask_map, cv::Size(kInputW, kInputH)); // 640x640
  cv::threshold(mask_map, mask_map, 128, 1, cv::THRESH_BINARY);


  /////////////////////////////////////////// Step3 ///////////////////////////////////////////
  cv::Mat mask_resized;
  cv::warpAffine(mask_map, mask_resized, IM, image.size(), cv::INTER_LINEAR);                                            
  //mask_resized = inverse_WarpAffine_CUDA(mask_map, (float *)affine_matrixs.i2d), image.size());


  /////////////////////////////////////////// Step4 ///////////////////////////////////////////
  // create color mask
  cv::Mat colored_mask = cv::Mat::ones(image.size(), CV_8UC3);
  colored_mask.setTo(color);

  cv::Mat masked_colored_mask;
  cv::bitwise_and(colored_mask, colored_mask, masked_colored_mask, mask_resized);

  // create mask indices
  cv::Mat mask_indices;
  cv::compare(mask_resized, 1, mask_indices, cv::CMP_EQ);
  
  cv::Mat image_masked, colored_mask_masked;
  image.copyTo(image_masked, mask_indices);
  masked_colored_mask.copyTo(colored_mask_masked, mask_indices);

  // weighted sum
  cv::Mat result_masked;
  cv::addWeighted(image_masked, 0.6, colored_mask_masked, 0.4, 0, result_masked);
  
  // copy result to image
  result_masked.copyTo(image, mask_indices);
  
}


static void draw_bbox(const cv::Mat& image, 
                      yolo::Box& obj, 
                      cv::Scalar& color,
                      const int kInputW,
                      const int kInputH)
{

#if use_threadpool //打开线程池
    std::lock_guard<std::mutex> lock(mtx_postprocess);
#endif

    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), color, 5);
    auto name    = cocolabels[obj.class_label];
    auto caption = cv::format("%s %.2f", name, obj.confidence);
    int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), color, -1);
    cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);

}


#if use_threadpool //打开线程池
cv::Mat yolov8_draw_box_segmant(yolo::BoxArray objs, 
                                const cv::Mat &image, 
                                const int kInputW,
                                const int kInputH)
{
  uint8_t b, g, r;
  try{
      threads::ThreadPool tp(16);
      std::vector<std::future<void>> v_mask; //分割部分
      std::vector<std::future<void>> v_bbox; //检测部分
      for (auto& obj : objs)
      {
        auto b_g_r = yolo::random_color(obj.class_label);
        b = std::get<0>(b_g_r);
        g = std::get<1>(b_g_r);
        r = std::get<2>(b_g_r);

        cv::Scalar color(b, g, r);

        auto ans_mask = tp.add(draw_mask,
                       image, 
                       obj, 
                       color,
                       kInputW,
                       kInputH,
                       affine_matrixs);

        auto ans_bbox = tp.add(draw_bbox,
                       image, 
                       obj, 
                       color,
                       kInputW,
                       kInputH);

        v_mask.push_back(std::move(ans_mask));
        v_bbox.push_back(std::move(ans_bbox));
      }

  }
  catch (std::exception& e){
      std::cout << "线程池出错：" << e.what() << std::endl;
  }

  //cv::imwrite(cv::format("output/image/%d_result.jpg", i++), image);
  return image;
}


#else //关闭线程池
cv::Mat yolov8_draw_box_segmant(yolo::BoxArray objs, 
                                const cv::Mat &image, 
                                const int kInputW,
                                const int kInputH,
                                AffineMatrix &affine_matrixs)
{
  static int i = 0;
  uint8_t b, g, r;

  for(auto& obj : objs){
    auto b_g_r = yolo::random_color(obj.class_label);
    b = std::get<0>(b_g_r);
    g = std::get<1>(b_g_r);
    r = std::get<2>(b_g_r);

    cv::Scalar color(b, g, r);

    draw_mask(image, obj, color, kInputW, kInputH, affine_matrixs);
    draw_bbox(image, obj, color, kInputW, kInputH);

  }

  //cv::imwrite(cv::format("output/image/%d_result.jpg", i++), image);
  return image;
}

#endif


