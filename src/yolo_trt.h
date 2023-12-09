#pragma once

#include "utils/config.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#include <future>


namespace yolo {

#define INFO(...) yolo::__log_func(__FILE__, __LINE__, __VA_ARGS__)
void __log_func(const char *file, int line, const char *fmt, ...);
  
class BaseMemory {
 public:
  BaseMemory() = default;
  BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);
  virtual ~BaseMemory();
  virtual void *gpu_realloc(size_t bytes);
  virtual void *cpu_realloc(size_t bytes);
  void release_gpu();
  void release_cpu();
  void release();
  inline bool owner_gpu() const { return owner_gpu_; }
  inline bool owner_cpu() const { return owner_cpu_; }
  inline size_t cpu_bytes() const { return cpu_bytes_; }
  inline size_t gpu_bytes() const { return gpu_bytes_; }
  virtual inline void *get_gpu() const { return gpu_; }
  virtual inline void *get_cpu() const { return cpu_; }
  void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

 protected:
  void *cpu_ = nullptr;
  size_t cpu_bytes_ = 0, cpu_capacity_ = 0;
  bool owner_cpu_ = true;

  void *gpu_ = nullptr;
  size_t gpu_bytes_ = 0, gpu_capacity_ = 0;
  bool owner_gpu_ = true;
};

template <typename _DT>
class Memory : public BaseMemory {
 public:
  Memory() = default;
  Memory(const Memory &other) = delete;
  Memory &operator=(const Memory &other) = delete;
  virtual _DT *gpu(size_t size) { return (_DT *)BaseMemory::gpu_realloc(size * sizeof(_DT)); }
  virtual _DT *cpu(size_t size) { return (_DT *)BaseMemory::cpu_realloc(size * sizeof(_DT)); }

  inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_DT); }
  inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_DT); }

  virtual inline _DT *gpu() const { return (_DT *)gpu_; }
  virtual inline _DT *cpu() const { return (_DT *)cpu_; }
};



struct InstanceSegmentMap {
  int width = 0, height = 0;      // width % 8 == 0
  unsigned char *data = nullptr;  // is width * height memory

  InstanceSegmentMap(int width, int height);
  virtual ~InstanceSegmentMap();
};

struct Box {
  float left, top, right, bottom, confidence;
  int class_label;
  std::shared_ptr<InstanceSegmentMap> seg;  // valid only in segment task

  Box() = default;
  Box(float left, float top, float right, float bottom, float confidence, int class_label)
      : left(left),top(top),right(right),bottom(bottom),confidence(confidence),class_label(class_label) {}
};

typedef std::vector<Box> BoxArray;

struct Image {
  const void *bgrptr = nullptr;
  int width = 0, height = 0;

  Image() = default;
  Image(const void *bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {}
};

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);

};  // namespace yolo