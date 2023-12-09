#include "draw.h"

void draw_matte(cv::Mat &img, cv::Mat &matte, const std::string format, const cv::Mat &bg)
{
    cv::resize(matte, matte, img.size()); // 将matte的大小变成了img的大小，参数分别是：输入图像，输出图像，输出图像的大小

    if (format == "foreground")
    {
        // 返回前景部分
        cv::Mat bg_temp;

        cv::cvtColor(matte, matte, cv::COLOR_GRAY2BGR); // 将matte_temp的通道数变成了3

        img.convertTo(img, CV_32FC3); // 将img的数据类型转换成CV_32FC3

        bg_temp.convertTo(bg_temp, CV_32FC3); // 将bg的数据类型转换成CV_32FC3

        cv::multiply(img, matte, img); // 将img和matte对应位置的元素相乘，结果保存在img中

        img.convertTo(img, CV_8UC3); // 将img的数据类型转换成CV_8UC3
    }
    else if (format == "matte")
    {
        // 返回matte，将数据拷贝到img中
        // matte 转为 3 通道
        cv::cvtColor(matte, matte, cv::COLOR_GRAY2BGR); // 将matte_temp的通道数变成了3
        // 乘以 255
        cv::multiply(matte, cv::Scalar(255.0/10, 255.0/10, 255.0/10), matte); // 将matte和(255, 255, 255)对应位置的元素相乘，结果保存在matte中
        // 转为 8UC3
        matte.convertTo(matte, CV_8UC3); // 将matte的数据类型转换成CV_8UC3
        // 将matte的数据拷贝到img中
        matte.copyTo(img);
    }
    else if (format == "background")
    {
        // 返回背景部分
        // 函数功能：将img中的前景部分保留下来，背景部分变成了bg
        cv::Mat bg_temp;

        cv::cvtColor(matte, matte, cv::COLOR_GRAY2BGR); // 将matte_temp的通道数变成了3

        img.convertTo(img, CV_32FC3); // 将img的数据类型转换成CV_32FC3

        cv::resize(bg, bg_temp, img.size());  // 将bg的大小变成了img的大小，参数分别是：输入图像，输出图像，输出图像的大小
        bg_temp.convertTo(bg_temp, CV_32FC3); // 将bg的数据类型转换成CV_32FC3

        cv::multiply(img, matte, img); // 将img和matte对应位置的元素相乘，结果保存在img中
        // 将img中的前景部分保留下来，背景部分变成了bg：img = img * matte + bg * (1 - matte)
        cv::multiply(bg_temp, cv::Scalar(1.0, 1.0, 1.0) - matte, bg_temp); // 将bg和matte对应位置的元素相乘，结果保存在bg中

        cv::add(img, bg_temp, img);  // 将img和bg对应位置的元素相加，结果保存在img中
        img.convertTo(img, CV_8UC3); // 将img的数据类型转换成CV_8UC3
    }
}