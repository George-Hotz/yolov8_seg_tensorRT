#pragma once
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "yolo_infer.h"
#include "yolo_trt.h"
#include "preprocess.h"
#include <opencv2/opencv.hpp>

void Release_Memory();
cv::Mat yolov8_draw_box_segmant(yolo::BoxArray objs, 
                                const cv::Mat &image);

yolo::BoxArray post_process(float *output_detect_buffer,
                            float *output_segmant_buffer,
                            const float kConfThresh, 
                            const float kNmsThresh,
                            const int kNumClass,
                            const int kInputW,
                            const int kInputH);


