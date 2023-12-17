#include <opencv2/dnn/all_layers.hpp>
#include "utils/postprocess.h"
#include "utils/preprocess.h"
#include "utils/config.h"
#include "yolo_infer.h"


AffineMatrix affine_matrixs_yolov8;
AffineMatrix affine_matrixs_lowlight;
memory_ctrl::Memory<uint8_t> preprocess_buffer_yolov8;
memory_ctrl::Memory<uint8_t> preprocess_buffer_lowlight;

namespace yolo {

///////////////////////////////////// YOLOV8 /////////////////////////////////////

YOLOV8::YOLOV8(const std::string &model_path){
  engine_.reset(new TrtEngine(model_path));
}

YOLOV8::~YOLOV8(){
  preprocess_buffer_yolov8.release();
}

cv::Mat YOLOV8::run(const cv::Mat &img){
  preprocess(img);
  doInference();
  return postprocess(img);
}

void YOLOV8::doInference(){
  engine_->doInference(cpu_pre_flag);
}

void YOLOV8::preprocess(const cv::Mat &img){
  if(cpu_pre_flag){
      Preprocess_cpu(img, kInputW, kInputH, (float *)engine_->getHostBuffer(kInputTensorName)); 
  }
  else{
      Preprocess_gpu(img, kInputW, kInputH, (float *)engine_->getDeviceBuffer(kInputTensorName),
                     preprocess_buffer_yolov8, affine_matrixs_yolov8);
  }
}

cv::Mat YOLOV8::postprocess(const cv::Mat &img){
  auto bboxs = post_process((float *)engine_->getDeviceBuffer(kOutputTensorName_Detect),  //(8400 * 116)
                            (float *)engine_->getDeviceBuffer(kOutputTensorName_Segmant), //(32 * 160 * 160)
                            preprocess_buffer_yolov8, affine_matrixs_yolov8,
                            kConfThresh, kNmsThresh, kNumClass, kInputW, kInputH);

  auto result_img = yolov8_draw_box_segmant(bboxs, img, kInputW, kInputH, 
                                            affine_matrixs_yolov8);
  return result_img;
}


///////////////////////////////////// Low Light /////////////////////////////////////

Lowlight::Lowlight(const std::string &model_path){
  engine_.reset(new TrtEngine(model_path));
}

Lowlight::~Lowlight(){
  preprocess_buffer_lowlight.release();
}

cv::Mat Lowlight::run(const cv::Mat &img){
  preprocess(img);
  doInference();
  return postprocess(img);
}

void Lowlight::doInference(){
  engine_->doInference(cpu_pre_flag);
}

void Lowlight::preprocess(const cv::Mat &img){
  if(cpu_pre_flag){
      Preprocess_cpu(img, kInputW, kInputH, (float *)engine_->getHostBuffer(kInputTensorName)); 
  }
  else{
      Preprocess_gpu(img, kInputW, kInputH, (float *)engine_->getDeviceBuffer(kInputTensorName),
                     preprocess_buffer_lowlight, affine_matrixs_lowlight);
  }
}

cv::Mat Lowlight::postprocess(const cv::Mat &img){ //lowlight的输入输出维度相同
  float *engine_out_gpu = (float *)engine_->getDeviceBuffer(kOutputTensorName);
  auto bgrbgrbgr = rrrgggbbb_2_bgrbgrbgr(engine_out_gpu, kOutputW, kOutputH, img); 
  return bgrbgrbgr;
}



InstanceSegmentMap::InstanceSegmentMap(int width, int height) {
  this->width = width;
  this->height = height;
  //CUDA_CHECK(cudaMallocHost(&this->data, width * height));
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&this->data), width * height));
}

InstanceSegmentMap::~InstanceSegmentMap() {
  if (this->data) {
    CUDA_CHECK(cudaFreeHost(this->data));
    this->data = nullptr;
  }
  this->width = 0;
  this->height = 0;
}


std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f * s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
    case 0:
      r = v, g = t, b = p;
      break;
    case 1:
      r = q, g = v, b = p;
      break;
    case 2:
      r = p, g = v, b = t;
      break;
    case 3:
      r = p, g = q, b = v;
      break;
    case 4:
      r = t, g = p, b = v;
      break;
    case 5:
      r = v, g = p, b = q;
      break;
    default:
      r = 1, g = 1, b = 1;
      break;
  }
  return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                         static_cast<uint8_t>(r * 255));
}


std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
  float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
  float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
  return hsv2bgr(h_plane, s_plane, 1);
}


} //namespace yolo 