# yolov8-seg的模型部署(C++ TensorRT)
- 轻松实现yolov8的TensorRT高性能推理
- 没有复杂的包装，没有耦合!

# Yolo-Demo示例
- 支持YoloV8-Segment模型
- 🚀 Cuda内核-检测头的box解码
- 🚀 Cuda内核-分割头的Mask解码

# Yolov8 网络框架
![](yolov8.png)

# Yolov8检测框结果
![](Result.jpg)

# 描述
- build.cu: 创建TensorRT引擎的封装(FP16或INT8)
- engine.cu: TensorRT引擎的封装
- yolo_infer.cpp: yolov8-seg推理的封装
- yolo_trt.cpp: CPU/GPU内存管理的封装

# 注意
- yolov8-seg模型(80类) 有两个输出头: 检测头 , 分割头
- 检测头:(batchsize * 116 * 8400)
- 分割头:(batchsize * 32 * 160 * 160)        
-
- 下面的维度是我们需要的(见 Step 1):
- 检测头:(batchsize * 8400 * 116)
- 分割头:(batchsize * 32 * 160 * 160) 

### Step1 转化Onnx模型
`python v8trans.py yolov8s-seg.onnx`

- yolov8s-seg.onnx 转化之前
![](yolov8_seg_before.png)

- yolov8s-seg.onnx 转化之后
![](yolov8_seg_after.png)

### Step2 cmake编译工程
`cmake -S . -B build`
### Step3 build工程
`cmake --build build`
### Step4: 转化模型引擎 
```bash
./build/build --onnx_file=yolov8s_seg.onnx
```
### Step5: Yolov8-seg 推理部署
```bash
./build/yolov8_seg --yolov8 weights/yolov8s_seg.engine --vid_dir videos/
```

# 参考
- [💡Tutorial: 1. C++ TensorRT High-performance deployments(恩培计算机视觉)](https://enpeicv.com/)
- [💕Video: 2. Instance segmentation and detection of YoloV8](https://www.bilibili.com/video/BV1SY4y1C7E2)
- [🌻github_repo: TensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)

