#include "yolo_infer.h"

#include "utils/preprocess.h"
#include "utils/config.h"

#include <opencv2/dnn/all_layers.hpp>



YOLOV8::YOLOV8(const std::string &model_path)
{
    engine_.reset(new TrtEngine(model_path));
}

cv::Mat YOLOV8::run(const cv::Mat &img)
{
    preprocess(img);
    doInference();
    return postprocess(img);
}

void YOLOV8::doInference()
{
    engine_->doInference(cpu_pre_flag);
}

void YOLOV8::preprocess(const cv::Mat &img)
{
    if(cpu_pre_flag){
        Preprocess_cpu(img, kInputW, kInputH, (float *)engine_->getHostBuffer(kInputTensorName)); 
    }
    else{
        Preprocess_gpu(img, kInputW, kInputH, (float *)engine_->getDeviceBuffer(kInputTensorName));
    }
}

cv::Mat YOLOV8::postprocess(const cv::Mat &img)
{
    auto bboxs = post_process((float *)engine_->getDeviceBuffer(kOutputTensorName_Detect),  //(8400 * 116)
                              (float *)engine_->getDeviceBuffer(kOutputTensorName_Segmant), //(32 * 160 * 160)
                              kConfThresh, kNmsThresh, kNumClass, kInputW, kInputH);

    auto result_img = yolov8_draw_box_segmant(bboxs, img);

    return result_img;
}


