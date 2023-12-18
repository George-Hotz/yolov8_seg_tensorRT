# Deployment of yolov8-seg(C++ TensorRT)
- Easily implement producer-consumer models for various tasks and perform high-performance inference
- No complex packaging, no coupling!

# For the Yolo-Demo
- YoloV8-Segment is supported
- 🚀 Cuda kernel Detect-BBox-decoding
- 🚀 Cuda kernel Segmant-Mask-decoding 

# Yolov8 Network framework
![](yolov8.png)

# The result of bbox
![](Result.jpg)

# Description
- build.cu: The encapsultion of the creating TensorRT engine (FP16 or INT8)
- engine.cu: The encapsultion of the TensorRT engine
- yolo_infer.cpp: The encapsultion of the yolov8-seg inference
- yolo_trt.cpp: The encapsultion of the Memory manage of cpu and gpu

# yolov8_seg.cpp

- This part implements a producer-consumer model, which uses the queue as a shared resource to store the data produced by the producer, and the consumer takes the data from the queue for consumption.In this model, the producer and consumer are two different threads that share the same queue.A global variable buffer is defined to represent the queue with a size of buffer_size set to 10.Access to queues is protected with the mutex lock buffer_mutex to ensure that producers and consumers cannot access queues at the same time.not_full and not_empty are condition variables used to block the producer thread when the queue is full and the consumer thread when the queue is empty.It adds the video sequence to the queue.

- Three threads:
- 1、readFrame()，is a producer thread used to read video files.The producer thread first tries to acquire the mutex and then waits on the condition variable not_full until the queue is no longer full.Once the queue is full, the producer thread adds the video frame to the queue and sends a signal to the condition variable not_empty, notifying the consumer thread that there is data available in the queue for consumption.
- 2、inference()，is a consumer thread used to reason video frames in the queue.It takes video frames from the queue and consumes them.The consumer thread also first tries to acquire the mutex and then waits on the condition variable not_empty until there is data available for consumption in the queue.Once data is available for consumption, the consumer thread removes the number from the queue and sends a signal to the condition variable not_full, notifying the producer thread that there is space in the queue to continue producing data.The consumer thread consists of two Ai models in series, namely yolov8 detection segmentation model and Zero_DCE low light compensation model. Zero_DCE determines whether the light is too low by detecting the average brightness of the image. If the light is lower than the threshold, the low light compensation is turned on.
- 3、postprocess()，is a post-processing thread, mainly used to write video files.

- In the main function (), we create three threads, one for the producer thread readFrame(), one for the consumer thread inference(), and one for the postprocessing thread postprocess() after inference.Wait for three more threads to finish executing, and use the join() function to wait for the thread to finish executing.

# Attention
- yolov8-seg model (80 classes) has two heads outputs: Bbox Head , Segmant head
- Bbox Head :(batchsize * 116 * 8400)
- Segmant head :(batchsize * 32 * 160 * 160)        
-
- What dimension we need is below(See Step 1):
- Bbox Head :(batchsize * 8400 * 116)
- Segmant head :(batchsize * 32 * 160 * 160) 

### Step1: Convert the Onxx model
`python v8trans.py yolov8s-seg.onnx`

- yolov8s-seg.onnx convert before
![](yolov8_seg_before.png)

- yolov8s-seg.onnx convert after
![](yolov8_seg_after.png)


### Step2: Compile the project
`cmake -S . -B build`
### Step3: Build the project
`cmake --build build`
### Step4: Convert engine model, 
```bash
./build/build --onnx_file=yolov8s_seg.onnx
```
### Step5: Yolov8-seg inference
```bash
./build/yolov8_seg --yolov8 weights/yolov8s_seg.engine --vid_dir videos/
```

# Reference
- [💡Tutorial: 1. C++ TensorRT High-performance deployments(恩培计算机视觉)](https://enpeicv.com/)
- [💕Video: 2. Instance segmentation and detection of YoloV8](https://www.bilibili.com/video/BV1SY4y1C7E2)
- [🌻github_repo: TensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)

