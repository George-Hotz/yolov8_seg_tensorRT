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
#include "yolo_trt.h"
#include "utils/config.h"
#include "utils/postprocess.h"

#define High 640
#define Width 640


struct AffineMatrix {
  float i2d[6];  // image to dst(network), 2x3 matrix
  float d2i[6];  // dst to image, 2x3 matrix

  void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {
    float scale_x = std::get<0>(to) / (float)std::get<0>(from);
    float scale_y = std::get<1>(to) / (float)std::get<1>(from);
    float scale = std::min(scale_x, scale_y);
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = -scale * (std::get<0>(from)) * 0.5 + (std::get<0>(to)) * 0.5 + scale * 0.5 - 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = -scale * (std::get<1>(from)) * 0.5 + (std::get<1>(to)) * 0.5 + scale * 0.5 - 0.5;

    double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
    D = D != 0. ? double(1.) / D : double(0.);
    double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
    double b1 = -A11 * i2d[2] - A12 * i2d[5];
    double b2 = -A21 * i2d[2] - A22 * i2d[5];

    d2i[0] = A11;
    d2i[1] = A12;
    d2i[2] = b1;
    d2i[3] = A21;
    d2i[4] = A22;
    d2i[5] = b2;
  }
};


class YOLOV8 {
public:
    YOLOV8(const std::string &model_path);
    ~YOLOV8() = default;
    cv::Mat run(const cv::Mat& img);
    void doInference();
    void preprocess(const cv::Mat& img);
    cv::Mat postprocess(const cv::Mat &img);

    const int kNumClass = 80;       //yolov8检测类别
    const float kNmsThresh = 0.7f;  //NMS阈值
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

