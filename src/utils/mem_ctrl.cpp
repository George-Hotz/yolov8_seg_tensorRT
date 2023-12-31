#include "mem_ctrl.h"
#include "utils/config.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <stdarg.h>

#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace memory_ctrl {

using namespace std;
using namespace nvinfer1;

#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)

#define Assert(op)                 \
  do {                             \
    bool cond = !(!(op));          \
    if (!cond) {                   \
      INFO("Assert failed, " #op); \
      abort();                     \
    }                              \
  } while (0)

#define Assertf(op, ...)                             \
  do {                                               \
    bool cond = !(!(op));                            \
    if (!cond) {                                     \
      INFO("Assert failed, " #op " : " __VA_ARGS__); \
      abort();                                       \
    }                                                \
  } while (0)


static string file_name(const string &path, bool include_suffix) {
  if (path.empty()) return "";

  int p = path.rfind('/');
  int e = path.rfind('\\');
  p = std::max(p, e);
  p += 1;

  // include suffix
  if (include_suffix) return path.substr(p);

  int u = path.rfind('.');
  if (u == -1) return path.substr(p);

  if (u <= p) u = path.size();
  return path.substr(p, u - p);
}

void __log_func(const char *file, int line, const char *fmt, ...) {
  va_list vl;
  va_start(vl, fmt);
  char buffer[2048];
  string filename = file_name(file, true);
  int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
  vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
  fprintf(stdout, "%s\n", buffer);
}


BaseMemory::BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes) {
  reference(cpu, cpu_bytes, gpu, gpu_bytes);
}

void BaseMemory::reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes) {
  release();

  if (cpu == nullptr || cpu_bytes == 0) {
    cpu = nullptr;
    cpu_bytes = 0;
  }

  if (gpu == nullptr || gpu_bytes == 0) {
    gpu = nullptr;
    gpu_bytes = 0;
  }

  this->cpu_ = cpu;
  this->cpu_capacity_ = cpu_bytes;
  this->cpu_bytes_ = cpu_bytes;
  this->gpu_ = gpu;
  this->gpu_capacity_ = gpu_bytes;
  this->gpu_bytes_ = gpu_bytes;

  this->owner_cpu_ = !(cpu && cpu_bytes > 0);
  this->owner_gpu_ = !(gpu && gpu_bytes > 0);
}

BaseMemory::~BaseMemory() { release(); }

void *BaseMemory::gpu_realloc(size_t bytes) {
  if (gpu_capacity_ < bytes) {
    release_gpu();

    gpu_capacity_ = bytes;
    checkRuntime(cudaMalloc(&gpu_, bytes));
    // checkRuntime(cudaMemset(gpu_, 0, size));
  }
  gpu_bytes_ = bytes;
  return gpu_;
}

void *BaseMemory::cpu_realloc(size_t bytes) {
  if (cpu_capacity_ < bytes) {
    release_cpu();

    cpu_capacity_ = bytes;
    checkRuntime(cudaMallocHost(&cpu_, bytes));
    Assert(cpu_ != nullptr);
    // memset(cpu_, 0, size);
  }
  cpu_bytes_ = bytes;
  return cpu_;
}

void BaseMemory::release_cpu() {
  if (cpu_) {
    if (owner_cpu_) {
      checkRuntime(cudaFreeHost(cpu_));
    }
    cpu_ = nullptr;
  }
  cpu_capacity_ = 0;
  cpu_bytes_ = 0;
}

void BaseMemory::release_gpu() {
  if (gpu_) {
    if (owner_gpu_) {
      checkRuntime(cudaFree(gpu_));
    }
    gpu_ = nullptr;
  }
  gpu_capacity_ = 0;
  gpu_bytes_ = 0;
}

void BaseMemory::release() {
  release_cpu();
  release_gpu();
}


}//namespace mem_ctrl