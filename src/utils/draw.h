#pragma once

#include <opencv2/opencv.hpp>

void draw_matte(cv::Mat &img, cv::Mat &matte, const std::string format, const cv::Mat &bg);