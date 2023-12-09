#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include "utils/config.h"
#include "yolo_infer.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK


void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();
void cuda_batch_preprocess(std::vector<cv::Mat> &img_batch,
                           float *dst, int dst_width, int dst_height);
void Preprocess_cpu(const cv::Mat& img, int inputW, int inputH, float *input_host_buffer);
void Preprocess_gpu(const cv::Mat &src, int inputW, int inputH, float *input_device_buffer);