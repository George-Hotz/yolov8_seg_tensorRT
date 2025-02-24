#include <iostream>
#include "gflags/gflags.h"
#include "src/utils/thread_pool.h"
#include "src/utils/preprocess.h"
#include "src/utils/postprocess.h"
#include "yolo_infer.h"
#include <glob.h>
#include <queue>
#include <mutex>
#include <vector>
#include <thread>
#include <fstream>
#include <condition_variable>

#define PRINT_STEP_TIME 0
#define PRINT_ALL_TIME 0
#define DEBUG 0

// 缓存大小
const int BUFFER_SIZE = 10;

// 每个阶段需要传递的缓存
std::queue<cv::Mat> stage_1_frame;
std::queue<cv::Mat> stage_2_frame;

// 每个阶段的互斥锁
std::mutex stage_1_mutex;
std::mutex stage_2_mutex;

// 每个阶段的not_full条件变量
std::condition_variable stage_1_not_full;
std::condition_variable stage_2_not_full;

// 每个阶段的not_empty条件变量
std::condition_variable stage_1_not_empty;
std::condition_variable stage_2_not_empty;


class Yolov8_Segment_App
{
public:

    ~Yolov8_Segment_App(){
        std::cout << "Yolov8_Segment_App destructor" << std::endl;
    }
 
    Yolov8_Segment_App(const yolo::YOLOV8 yolov8, 
                       const yolo::Lowlight lowlight,
                       const std::string &input_video_path)
        : yolov8{yolov8}, lowlight{lowlight}
    {
        
        std::cout << "当前使用的是视频文件" << std::endl;
        cap = cv::VideoCapture(input_video_path);
        
        // 获取画面尺寸
        frameSize_ = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), 
                              cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        // 获取帧率
        video_fps_ = cap.get(cv::CAP_PROP_FPS);

        writer.open("./output/video/yolov8_output.mp4", 
                    cv::VideoWriter::fourcc('H', '2', '6', '4'), 
                    video_fps_, 
                    frameSize_);

        std::cout << "width: " << frameSize_.width 
                  << " height: " << frameSize_.height 
                  << " fps: " << video_fps_ << std::endl;
    };

    // read frame
    void readFrame()
    {
        std::cout << "线程1启动" << std::endl;

        cv::Mat frame;
        while (cap.isOpened())
        {
            // step1 start
            auto start_1 = std::chrono::high_resolution_clock::now();
            cap >> frame;
            if (frame.empty())
            {
                cap.release();
                std::cout << "视频读取完毕" << std::endl;
                std::cout << "线程1退出" << std::endl;
                file_processed_done = true;
                break;
            }
            // step1 end
            auto end_1 = std::chrono::high_resolution_clock::now();
            auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count() / 1000.f;

#if PRINT_STEP_TIME
            std::cout << "step1: " << elapsed_1 << "ms"
                      << ", fps: " << 1000.f / elapsed_1 << std::endl;
#endif

            {
                // 互斥锁
                std::unique_lock<std::mutex> lock(stage_1_mutex);
                // 如果缓存满了，就等待 (wait用法：stage_1_frame.size() < BUFFER_SIZE则执行)
                stage_1_not_full.wait(lock, []
                                    { return stage_1_frame.size() < BUFFER_SIZE; });
                // 增加一个元素
                stage_1_frame.push(frame);
#if DEBUG
                std::cout<< "线程1,缓存push后stage_1_frame剩余大小:" << stage_1_frame.size() <<std::endl;
#endif
                // 通知下一个线程可以开始了
                stage_1_not_empty.notify_one();
            }
        }
    }
    
    // 推理
    void inference()
    {
        std::cout << "线程2启动" << std::endl;
        
        cv::Mat frame;
        while (true)
        {
            // 检查是否退出
            if (file_processed_done && stage_1_frame.empty()){
                std::cout << "线程2退出" << std::endl;
                break;
            }
            // 使用{} 限制作用域，否则锁会在一次循环结束后才释放
            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock(stage_1_mutex);
                // 如果缓存为空，就等待 (wait用法：!stage_1_frame.empty();为真则执行)
                stage_1_not_empty.wait(lock, []
                                       { return !stage_1_frame.empty(); });
                // 取出一个元素
                frame = stage_1_frame.front();
                stage_1_frame.pop();
#if DEBUG
                std::cout<< "线程2,缓存pop后stage_1_frame剩余大小:" << stage_1_frame.size() <<std::endl;
#endif
                // 通知上一个线程可以开始了
                stage_1_not_full.notify_one();
            }

            
            // step2 start
            auto start_2 = std::chrono::high_resolution_clock::now();

            //使用低光照增强模型
            if(use_lowlight_enhance)
            {
                //光照过低则需要补偿
                if(check_Brightness(frame, threshold)){
                    frame = lowlight.run(frame);
                }
            }

            // ========== 执行推理 =========
            auto output = yolov8.run(frame);

            // step2 end
            auto end_2 = std::chrono::high_resolution_clock::now();
            elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count() / 1000.f;

#if PRINT_STEP_TIME
            std::cout << "step2: " << elapsed_2 << "ms"
                      << ", fps: " << 1000.f / elapsed_2 << std::endl;
#endif
            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock2(stage_2_mutex);
                // not full
                stage_2_not_full.wait(lock2, []
                                      { return stage_2_frame.size() < BUFFER_SIZE; });
                //   push
                stage_2_frame.push(frame);
#if DEBUG
                std::cout<< "线程2,缓存push后stage_2_frame剩余大小:" << stage_2_frame.size() <<std::endl;
#endif
                // not empty
                stage_2_not_empty.notify_one();
            }
        }
    }

    

    void postprocess()
    {
        std::cout << "线程3启动" << std::endl;
        cv::Mat frame;

        // 记录开始时间
        auto start_all = std::chrono::high_resolution_clock::now();
        int frame_count = 0;

        while (true)
        {
            // 检查是否退出
            if (file_processed_done && stage_2_frame.empty())
            {
                writer.release();
                std::cout << "线程3退出" << std::endl;
                break;
            }

            {
                // 使用互斥锁
                std::unique_lock<std::mutex> lock(stage_2_mutex);
                // 如果缓存为空，就等待
                stage_2_not_empty.wait(lock, []
                                       { return !stage_2_frame.empty(); });
                // 取出一个元素
                frame = stage_2_frame.front();
                stage_2_frame.pop();
#if DEBUG
                std::cout<< "线程3,缓存pop后stage_2_frame剩余大小:" << stage_2_frame.size() <<std::endl;
#endif
                // 通知上一个线程可以开始了
                stage_2_not_full.notify_one();
            }


            std::string fps_str = "FPS: " + std::to_string(1000.f / elapsed_2);
            cv::putText(frame, fps_str, cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

            // 写入视频文件
            writer.write(frame);

#if PRINT_ALL_TIME
            frame_count++;
            // all end
            auto end_all = std::chrono::high_resolution_clock::now();
            auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
            // 每隔1秒打印一次
            if (elapsed_all_2 > 1000)
            {
                std::cout << "method 2 all steps time(ms): " << elapsed_all_2 
                          << ", fps: " << frame_count / (elapsed_all_2 / 1000.0f) 
                          << ",frame count: " << frame_count << std::endl;
                frame_count = 0;
                start_all = std::chrono::high_resolution_clock::now();
            }
#endif
            
        }
    }

private:

    yolo::YOLOV8 yolov8;
    yolo::Lowlight lowlight;
    uint8_t threshold = 35;           // 低光照阈值
    cv::VideoCapture cap;             // 视频流
    cv::VideoWriter writer;
    cv::Size frameSize_;              // 视频帧大小
    float video_fps_;                 // 视频帧率
    float elapsed_1 = 0;              // inference time
    float elapsed_2 = 0;              // inference time
    bool file_processed_done = false; // 文件处理完成标志
    bool use_lowlight_enhance=true;   //默认使用低光照增强模型
};



DEFINE_string(yolov8, "yolov8s-seg.engine", "model path");
DEFINE_string(lowlight, "Zero_DCE.engine", "model path");
DEFINE_string(input_dir, "", "input files directory");

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "用法: " << argv[0] << " --yolov8=path/to/engine --vid_dir=videos" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string yolov8_path = FLAGS_yolov8;       // yolov8检测分割模型
    std::string lowlight_path = FLAGS_lowlight;   // Zero_DCE低光照补偿模型
    std::string input_dir = FLAGS_input_dir;      // 输入视频文件


    auto yolov8 = yolo::YOLOV8(yolov8_path);
    auto lowlight = yolo::Lowlight(lowlight_path);

    auto app = Yolov8_Segment_App(yolov8, lowlight, input_dir);

    // thread 1 : read video stream
    std::thread T_readFrame(&Yolov8_Segment_App::readFrame, &app);
    // thread 2: inference
    std::thread T_inference(&Yolov8_Segment_App::inference, &app);
    // thread 3: postprocess
    std::thread T_postprocess(&Yolov8_Segment_App::postprocess, &app);

    // 等待线程结束
    T_readFrame.join();
    T_inference.join();
    T_postprocess.join();

    //Release_Memory();

    return 0;
}

