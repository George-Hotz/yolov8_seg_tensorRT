#pragma once
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "yolo_infer.h"

cv::Mat rrrgggbbb_2_bgrbgrbgr(float* engine_out_gpu, 
                              const int output_w,
                              const int output_h,
                              const cv::Mat &img);
                              
cv::Mat yolov8_draw_box_segmant(yolo::BoxArray objs, 
                                const cv::Mat &image, 
                                const int kInputW,
                                const int kInputH,
                                AffineMatrix &affine_matrixs);

yolo::BoxArray post_process(float *output_detect_buffer_,   //Network输出检测头
                            float *output_segmant_buffer_,  //Network输出分割头
                            memory_ctrl::Memory<uint8_t> &pre_buffer,
                            AffineMatrix &affine_matrixs,
                            const float kConfThresh, 
                            const float kNmsThresh,
                            const int kNumClass,
                            const int kInputW,
                            const int kInputH);


