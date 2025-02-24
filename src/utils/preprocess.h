#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include "utils/config.h"
#include "utils/mem_ctrl.h"

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

// typedef struct{
//     memory_ctrl::Memory<uint8_t> preprocess_buffer; //预处理相关的内存申请
//     AffineMatrix affine_matrixs;             //仿射变换矩阵
// }preprocess_manager;


void Preprocess_cpu(const cv::Mat& img, int inputW, int inputH, float *input_host_buffer);
void Preprocess_gpu(const cv::Mat &src, int inputW, int inputH, float *input_device_buffer,
                    memory_ctrl::Memory<uint8_t> &preprocess_buffer, AffineMatrix &affine_matrixs);
bool check_Brightness(const cv::Mat &input, const uint8_t threshold);