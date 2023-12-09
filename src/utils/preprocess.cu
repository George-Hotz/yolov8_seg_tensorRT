#include "preprocess.h"
#include "yolo_trt.h"

yolo::Memory<uint8_t> preprocess_buffer; //预处理相关的内存申请
AffineMatrix affine_matrixs;            //仿射变换矩阵

//对齐操作，对齐数：align
inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; } 

__global__ void warpaffine_kernel(
    uint8_t *src, int src_line_size, int src_width, int src_height, 
    float *dst, int dst_width, int dst_height, uint8_t const_value_st,
    float * warp_affine_matrix_2_3)
{
  float mean[] {0.485, 0.456, 0.406};
  float std[] {0.229, 0.224, 0.225};

  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= dst_width || dy >= dst_height) return;

  float m_x1 = warp_affine_matrix_2_3[0];
  float m_y1 = warp_affine_matrix_2_3[1];
  float m_z1 = warp_affine_matrix_2_3[2];
  float m_x2 = warp_affine_matrix_2_3[3];
  float m_y2 = warp_affine_matrix_2_3[4];
  float m_z2 = warp_affine_matrix_2_3[5];

  float src_x = m_x1 * dx + m_y1 * dy + m_z1;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    // out of range
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  } else {
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t *v1 = const_value;
    uint8_t *v2 = const_value;
    uint8_t *v3 = const_value;
    uint8_t *v4 = const_value;
    if (y_low >= 0) {
      if (x_low >= 0) 
        v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width) 
        v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height) {
      if (x_low >= 0) 
        v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width) 
        v4 = src + y_high * src_line_size + x_high * 3;
    }

    // same to opencv
    c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
    c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
    c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
  }

  float t = c2;
  c2 = c0;
  c0 = t;

  // normalization
  c0 = c0 / 255.0f;
  c1 = c1 / 255.0f;
  c2 = c2 / 255.0f;

  //imagenet normalization
  // c0 = (c0-mean[0])/std[0];
  // c1 = (c1-mean[1])/std[1];
  // c2 = (c2-mean[2])/std[2];

  // rgbrgbrgb to rrrgggbbb
  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;

}


void cuda_preprocess(
    uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height)
{

  // 计算变换矩阵
  affine_matrixs.compute(std::make_tuple(src_width, src_height),
                         std::make_tuple(dst_width, dst_height));

  size_t size_matrix = upbound(sizeof(affine_matrixs.d2i), 32);
  size_t size_image = src_width * src_height * 3;

  //GPU申请内存
  uint8_t *gpu_workspace = preprocess_buffer.gpu(size_matrix + size_image);
  float *affine_matrix_device = (float *)gpu_workspace;
  uint8_t *img_buffer_device = gpu_workspace + size_matrix;

  //CPU申请内存
  uint8_t *cpu_workspace = preprocess_buffer.cpu(size_matrix + size_image);
  float *affine_matrix_host = (float *)cpu_workspace;
  uint8_t *img_buffer_host = cpu_workspace + size_matrix;

  //对申请的CPU内存赋值
  memcpy(affine_matrix_host, affine_matrixs.d2i, size_matrix);//赋值affine_matrixs的数据
  memcpy(img_buffer_host, src, size_image);  //赋值src的数据

  cudaStream_t stream_ = nullptr;  //加速

  // CPU-->GPU转移内存
  CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, size_image, 
                                cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, size_matrix,
                                cudaMemcpyHostToDevice, stream_));

  dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
  dim3 block(32, 32);

  // 调用kernel函数
  warpaffine_kernel<<<grid, block, 0, stream_>>>(
      img_buffer_device, src_width * 3, src_width,
      src_height, dst, dst_width, dst_height, 0, 
      affine_matrix_device);
}


// 使用cuda预处理所有步骤
void Preprocess_gpu(const cv::Mat &src, int inputW, int inputH, float *input_device_buffer)
{
  
  cuda_preprocess((uint8_t *)src.ptr(), src.cols, src.rows, input_device_buffer, inputW, inputH);
}


////////////////////////////////  CPU Part  ////////////////////////////////

void Preprocess_cpu(const cv::Mat &img, int inputW, int inputH, float *input_host_buffer)
{

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(inputW, inputH));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32FC3);
    cv::subtract(normalized, cv::Scalar(127.5, 127.5, 127.5), normalized);
    cv::divide(normalized, cv::Scalar(127.5, 127.5, 127.5), normalized);
    // split it into three channels
    std::vector<cv::Mat> nchw_channels;
    cv::split(normalized, nchw_channels);

    for (auto &img : nchw_channels)
    {
        img = img.reshape(1, 1);
    }

    cv::Mat nchw;
    cv::hconcat(nchw_channels, nchw);

    memcpy(input_host_buffer, nchw.data, 3 * inputH * inputW * sizeof(float));
}
