#include "postprocess.h"
#include "config.h"

#define GPU_BLOCK_THREADS 512



//该情况假设batchsize=1,  (116 = 4 + 80 + 32)
const int det_head_dim[3] = {1, 8400, 116};       //检测头的维度
const int seg_head_dim[4] = {1, 32, 160, 160};    //分割头的维度

yolo::Memory<float> Output_box_array;           //box结果相关的内存申请 
yolo::Memory<float> Output_detect_buffer;       //检测头相关的内存申请 (8400 * 116)
yolo::Memory<float> Output_segmant_buffer;      //分割头相关的内存申请 (32 * 160 * 160)
std::vector<std::shared_ptr<yolo::Memory<unsigned char>>> box_segment_cache;

extern yolo::Memory<uint8_t> preprocess_buffer; //预处理相关的内存申请 (affine matrix)
extern AffineMatrix affine_matrixs; 

const int NUM_BOX_ELEMENT = 8;  // left, top, right, bottom, confidence, class, keepflag, row_index(output)
const int MAX_IMAGE_BOXES = 1024;


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
                                          int mask_height, unsigned char *mask_out,
                                          int mask_dim, int out_width, int out_height) {
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
    float wval = mask_weights[ic];
    cumprod += cval * wval;
  }

  float alpha = 1.0f / (1.0f + exp(-cumprod));
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
                               int mask_width, int mask_height, unsigned char *mask_out,
                               int mask_dim, int out_width, int out_height, cudaStream_t stream) {
  // mask_weights is mask_dim(32 element) gpu pointer
  dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
  dim3 block(32, 32);

  decode_single_mask_kernel<<<grid, block, 0, stream>>>(
                                          left, top, mask_weights, mask_predict, 
                                          mask_width, mask_height, mask_out, mask_dim, 
                                          out_width, out_height);
}


yolo::BoxArray post_process(float *output_detect_buffer,
                            float *output_segmant_buffer,
                            const float kConfThresh, 
                            const float kNmsThresh,
                            const int kNumClass,
                            const int kInputW,
                            const int kInputH){
  
  size_t size_detect  = det_head_dim[1] * det_head_dim[2] * sizeof(float);
  size_t size_segmant = seg_head_dim[1] * seg_head_dim[2] * seg_head_dim[3] * sizeof(float);
  size_t size_bbox_array = (MAX_IMAGE_BOXES * NUM_BOX_ELEMENT + 32) * sizeof(float);

  //申请内存
  float *detect_device = Output_detect_buffer.gpu(size_detect);      //GPU端的检测buff (8400 * 116)
  float *segmant_device = Output_segmant_buffer.gpu(size_segmant);   //GPU端的分割buff (32 * 160 * 160)
  float *box_array_device = Output_box_array.gpu(size_bbox_array);   //GPU端的box_array
  float *box_array_host = Output_box_array.cpu(size_bbox_array);     //CPU端的box_array

  float *affine_matrix_device = (float *)preprocess_buffer.gpu();

  cudaStream_t stream_ = nullptr;  //加速

  // 内存初始化
  CUDA_CHECK(cudaMemsetAsync(box_array_device, 0, sizeof(int), stream_));

  // GPU-->GPU转移内存
  CUDA_CHECK(cudaMemcpyAsync(detect_device, output_detect_buffer, size_detect, 
                                cudaMemcpyDeviceToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(segmant_device, output_segmant_buffer, size_segmant,
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
      yolo::Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label); //构造函数来初始化box结果
      
      //分割部分
      int row_index = pbox[7];
      int mask_dim = seg_head_dim[1];
      float *mask_weights = detect_device + 
                            row_index * det_head_dim[2] + 
                            kNumClass + 4;

      float *mask_head_predict = Output_segmant_buffer.gpu();
      float left, top, right, bottom;
      float *i2d = affine_matrixs.i2d;
      affine_project(i2d, pbox[0], pbox[1], &left, &top);
      affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

      float box_width = right - left;
      float box_height = bottom - top;

      float scale_to_predict_x = seg_head_dim[3] / (float)kInputW;
      float scale_to_predict_y = seg_head_dim[2] / (float)kInputH;
      int mask_out_width = box_width * scale_to_predict_x + 0.5f;
      int mask_out_height = box_height * scale_to_predict_y + 0.5f;

      if (mask_out_width > 0 && mask_out_height > 0) {
        if (imemory >= (int)box_segment_cache.size()) {
          box_segment_cache.push_back(std::make_shared<yolo::Memory<unsigned char>>());
        }

        int bytes_of_mask_out = mask_out_width * mask_out_height;
        auto box_segment_output_memory = box_segment_cache[imemory];
        result_object_box.seg =
            std::make_shared<yolo::InstanceSegmentMap>(mask_out_width, mask_out_height);

        unsigned char *mask_out_device = box_segment_output_memory->gpu(bytes_of_mask_out);
        unsigned char *mask_out_host = result_object_box.seg->data;
        decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
                            mask_head_predict + seg_head_dim[2] * seg_head_dim[3],
                            seg_head_dim[3], seg_head_dim[2], mask_out_device,
                            mask_dim, mask_out_width, mask_out_height, stream_);
        CUDA_CHECK(cudaMemcpyAsync(mask_out_host, mask_out_device,
                                    box_segment_output_memory->gpu_bytes(),
                                    cudaMemcpyDeviceToHost, stream_));
      }
      
      output.emplace_back(result_object_box);
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream_));

  return arrout;
}

//释放内存！！！
void Release_Memory(){

    preprocess_buffer.release();
    Output_box_array.release();           
    Output_detect_buffer.release();    
    Output_segmant_buffer.release();
    for(int i=0;i<box_segment_cache.size();i++)
        box_segment_cache[i]->release();
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


cv::Mat yolov8_draw_box_segmant(yolo::BoxArray objs, const cv::Mat &image){

  int i = 0;
  for (auto &obj : objs) {
    uint8_t b, g, r;
    auto b_g_r = yolo::random_color(obj.class_label);
    b = std::get<0>(b_g_r);
    g = std::get<1>(b_g_r);
    r = std::get<2>(b_g_r);

    //tie(b, g, r) = yolo::random_color(obj.class_label);
    cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                  cv::Scalar(b, g, r), 5);

    auto name = cocolabels[obj.class_label];
    auto caption = cv::format("%s %.2f", name, obj.confidence);
    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                  cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
    cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);

    // if (obj.seg) {
    //   cv::imwrite(cv::format("%d_mask.jpg", i),
    //               cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
    //   i++;
    // }
  }

  // printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
  // cv::imwrite("Result.jpg", image);

  return image;
}




