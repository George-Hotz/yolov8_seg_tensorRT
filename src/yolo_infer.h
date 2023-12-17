#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <iostream>  
#include <initializer_list>
#include <memory>
#include <future>

#include <opencv2/opencv.hpp>

#include "engine.h"
#include "utils/config.h"
#include "utils/preprocess.h"

#define High 640
#define Width 640

namespace yolo {

class YOLOV8 {
public:
  YOLOV8(const std::string &model_path);
  ~YOLOV8();

  void doInference();
  void preprocess(const cv::Mat& img);
  cv::Mat run(const cv::Mat& img);
  cv::Mat postprocess(const cv::Mat &img);

  const int kNumClass = 80;       //yolov8检测类别
  const float kNmsThresh = 0.6f;  //NMS阈值
  const float kConfThresh = 0.4f; //置信度阈值

private:
  std::shared_ptr<TrtEngine> engine_;
  bool cpu_pre_flag = false;
  const int kInputH = High;
  const int kInputW = Width;
  const int kInputC = 3;
  const int kOutputH = High;
  const int kOutputW = Width;
  const char* kInputTensorName = "images";
  const char* kOutputTensorName_Detect = "output0";  //(8400 * 116)
  const char* kOutputTensorName_Segmant = "output1"; //(32 * 160 * 160)
};


class Lowlight {
public:
  Lowlight(const std::string &model_path);
  ~Lowlight();

  void doInference();
  void preprocess(const cv::Mat& img);
  cv::Mat run(const cv::Mat& img);
  cv::Mat postprocess(const cv::Mat &img);

private:
  std::shared_ptr<TrtEngine> engine_;
  bool cpu_pre_flag = false;
  const int kInputH = High;
  const int kInputW = Width;
  const int kInputC = 3;
  const int kOutputH = High;
  const int kOutputW = Width;
  const char* kInputTensorName = "input";
  const char* kOutputTensorName = "output";  
};


struct InstanceSegmentMap {
  int width = 0, height = 0;      // width % 8 == 0
  unsigned char *data = nullptr;  // is width * height memory

  InstanceSegmentMap(int width, int height);
  virtual ~InstanceSegmentMap();
};

struct Box {
  float left, top, right, bottom, confidence;
  int class_label;
  std::shared_ptr<InstanceSegmentMap> seg;  // valid only in segment task

  Box() = default;
  Box(float left, float top, float right, float bottom, float confidence, int class_label)
      : left(left),top(top),right(right),bottom(bottom),confidence(confidence),class_label(class_label) {}
};

typedef std::vector<Box> BoxArray;

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);


} //namespace yolo

