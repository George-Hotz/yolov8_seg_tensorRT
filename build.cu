#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "safeCommon.h"
#include "utils/preprocess.h"
#include "yolo_infer.h"
#include "gflags/gflags.h"

#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using NetPreprocessor = void (*)(const cv::Mat &, int, int, void *);

// 定义校准数据读取器
class CalibrationDataReader : public IInt8MinMaxCalibrator
{
public:
    CalibrationDataReader(
        const std::string &dataDir,
        const std::string &list,
        const char *inputName,
        int inputH, int inputW,
        void (*preprecess)(const cv::Mat &, int, int, float *),
        int batchSize = 1) : mDataDir(dataDir),
                             mCacheFileName("calibration.cache"),
                             mInputName(inputName),
                             mBatchSize(batchSize), mImgSize(inputH * inputW),
                             mInputH(inputH), mInputW(inputW),
                             mPreprocess(preprecess)
    {
        mInputDims = {4, 1, 3, inputH, inputW}; // 根据网络输入尺寸设置
        std::cout << "CalibrationDataReader: " << mInputDims.d[0] << " " << mInputDims.d[1] << " " << mInputDims.d[2] << " " << mInputDims.d[3] << std::endl;
        std::cout << "nbDims: " << mInputDims.nbDims << std::endl;
        mInputCount = mBatchSize * samplesCommon::volume(mInputDims);
        std::cout << "CalibrationDataReader: " << mInputCount << std::endl;
        // load file names from list
        cudaMalloc(&mDeviceBatchData, mInputCount * sizeof(float));
        cudaMallocHost(&mHostBatchData, mInputCount * sizeof(float));
        std::ifstream infile(list);
        std::string line;
        while (std::getline(infile, line))
        {
            sample::gLogInfo << line << std::endl;
            mFileNames.push_back(line);
        }
        mBatchCount = mFileNames.size() / mBatchSize;
        std::cout << "CalibrationDataReader: " << mFileNames.size() << " images, " << mBatchCount << " batches." << std::endl;
    }

    int32_t getBatchSize() const noexcept override
    {
        return mBatchSize;
    }

    bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override
    {
        if (mCurBatch + 1 > mBatchCount)
        {
            return false;
        }

        int offset = mInputW * mInputH * 3 * sizeof(float);
        for (int i = 0; i < mBatchSize; i++)
        {
            int idx = mCurBatch * mBatchSize + i;
            std::string fileName = mDataDir + "/" + mFileNames[idx];
            cv::Mat img = cv::imread(fileName);
            if (img.empty())
            {
                std::cout << "read image failed: " << fileName << std::endl;
                return false;
            }

            mPreprocess(img, mInputW, mInputH, mHostBatchData + i * offset);
            cudaMemcpy(mDeviceBatchData + i * offset, mHostBatchData + i * offset, offset, cudaMemcpyHostToDevice);
        }
        for (int i = 0; i < nbBindings; i++)
        {
            if (!strcmp(names[i], mInputName))
            {
                bindings[i] = mDeviceBatchData + i * offset;
            }
        }

        mCurBatch++;
        return true;
    }

    const void *readCalibrationCache(std::size_t &length) noexcept override
    {
        // read from file
        mCalibrationCache.clear();

        std::ifstream input(mCacheFileName, std::ios::binary);
        input >> std::noskipws;

        if (input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(mCalibrationCache));
        }

        length = mCalibrationCache.size();

        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void *cache, std::size_t length) noexcept override
    {
        // write tensorrt calibration cache to file
        std::ofstream output(mCacheFileName, std::ios::binary);
        output.write(reinterpret_cast<const char *>(cache), length);
    }

private:
    std::string mDataDir;
    std::string mCacheFileName;
    std::vector<std::string> mFileNames;
    int mBatchSize;
    Dims mInputDims;
    int mInputCount;
    float *mDeviceBatchData{nullptr};
    float *mHostBatchData{nullptr};
    int mBatchCount;
    const char *mInputName;
    int mImgSize;
    int mInputH;
    int mInputW;
    int mCurBatch{0};
    std::vector<char> mCalibrationCache;
    void (*mPreprocess)(const cv::Mat &, int, int, float *);
};

// define input flags


DEFINE_string(onnx_file, "mitb1.onnx", "onnx file path");
DEFINE_string(calib_dir, "", "calibration data dir");
DEFINE_string(calib_list_file, "", "calibration data list file");
DEFINE_string(input_name, "intput", "network input name");
DEFINE_int32(input_h, High, "network input height");
DEFINE_int32(input_w, Width, "network input width");
DEFINE_int32(input_c, 3, "network input channel");
DEFINE_bool(int8, false, "use int8 mode");
DEFINE_bool(dla, false, "use dla 0");
DEFINE_string(model_name, "mitb1", "model name");
DEFINE_string(format, "nchw", "input format");

template <typename T>
using TrtUniquePtr = samplesCommon::SampleUniquePtr<T>;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        // std::cout << "Usage: ./build [onnx_file] [calib_dir] [calib_list_file]" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const char *onnx_file_path = FLAGS_onnx_file.c_str();
    const char *calib_dir = FLAGS_calib_dir.c_str();
    const char *calib_list_file = FLAGS_calib_list_file.c_str();

    int input_h = FLAGS_input_h;
    int input_w = FLAGS_input_w;
    const char *input_name = FLAGS_input_name.c_str();

    bool useInt8 = FLAGS_int8;
    bool useDLA = FLAGS_dla;
    
    // remove extension of onnx_file_path
    std::string output_file_name = FLAGS_onnx_file.substr(0, FLAGS_onnx_file.find_last_of(".")) + ".engine";

    sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);

    // 1. Create builder
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return -1;
    }

    // 2. Create network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return -1;
    }

    // 3. Create builder config
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return -1;
    }

    // 4. Create ONNX Parser
    auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));

    // 5. Parse ONNX model
    auto parsed = parser->parseFromFile(onnx_file_path, static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return -1;
    }

    auto input = network->getInput(0);
    auto profile = builder->createOptimizationProfile();
    if (FLAGS_format == "nchw")
    {
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, FLAGS_input_c, input_h, input_w});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, FLAGS_input_c, input_h, input_w});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, FLAGS_input_c, input_h, input_w});
    }
    else
    {
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, input_h, input_w, FLAGS_input_c});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, input_h, input_w, FLAGS_input_c});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, input_h, input_w, FLAGS_input_c});
    }
    config->addOptimizationProfile(profile);

    // 6. set calibration configuration
    if (!useInt8)
    {
        // if (true) {
        sample::gLogInfo << "using fp16 mode" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else
    {
        sample::gLogInfo << "using int8 mode" << std::endl;
        auto calibrator = new CalibrationDataReader(calib_dir, calib_list_file, input_name, input_h, input_w, Preprocess_cpu);
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator);
    }

    if (useDLA)
    {
        //enable GPUFallback mode:
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(0);

    }

    builder->setMaxBatchSize(1);

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

    // 7. Create CUDA stream for profiling
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return -1;
    }
    config->setProfileStream(*profileStream);

    // 8. Build Serialized Engine
    auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        return -1;
    }

    // 9. save engine
    std::ofstream engine_file(output_file_name, std::ios::binary);
    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    return 0;
}
