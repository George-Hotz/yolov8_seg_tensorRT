#include <iostream>

#include "gflags/gflags.h"

#include "src/utils/preprocess.h"
#include "utils/draw.h"
#include "yolo_infer.h"
#include <glob.h>
#include <vector>
using std::vector;

DEFINE_string(yolov8, "yolov8s-seg.engine", "model path");
DEFINE_string(vid_dir, "", "video files directory");

vector<std::string> globVector(const std::string &pattern)
{
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    vector<std::string> files;
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i)
    {
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

void Process(std::string filename, YOLOV8 yolov8)
{
    cv::VideoCapture cap;
    cap.open(filename);

    // get height with from cap
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "width: " << width << ", height: " << height << ", fps: " << fps << std::endl;

    // h264 save mp4
    cv::VideoWriter writer;
    // ./videos/test1.mp4 -> test
    std::string out_filename = filename.substr(filename.find_last_of("/") + 1);
    out_filename = out_filename.substr(0, out_filename.find_last_of("."));
    out_filename = "./output/" + out_filename + "_out.mp4";

    writer.open(out_filename, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));

    cv::Mat frame;

    int frame_id = 0;
    float sum_elapse = 0;
    while (cap.read(frame))
    {
        frame_id++;
        // FPS开始时间
        auto start = std::chrono::high_resolution_clock::now();

        auto output = yolov8.run(frame);

        // FPS结束时间
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;

        sum_elapse+=elapsed;

        // std::cout << "elapsed: " << elapsed << "ms" << std::endl;
        std::cout << "filename: " << filename << ", fps: " << 1000.f / elapsed << std::endl;
        std::string fps_str = "FPS: " + std::to_string(1000.f / elapsed);
        cv::putText(frame, fps_str, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

        writer.write(frame);
    }

    std::cout << " fps: " << 1000.f / (sum_elapse/frame_id) << std::endl;

    //release memory
    Release_Memory();
    // release writer
    writer.release();
    // release capture
    cap.release();

    std::cout << "process: " << filename << " done, output: " << out_filename << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        // gflags print help message
        std::cout << "Usage: " << argv[0] << " --yolov8=/weight/yolov8s-seg.engine --vid_dir=videos" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string yolo_path = FLAGS_yolov8;
    std::string vid_dir = FLAGS_vid_dir;

    auto yolov8 = YOLOV8(yolo_path);

    // list all files in directory
    vector<std::string> files = globVector(vid_dir + "/*.mp4");
    for (auto &file : files)
    {
        std::cout << file << std::endl;
        Process(file, yolov8);
    }

    return 0;
}