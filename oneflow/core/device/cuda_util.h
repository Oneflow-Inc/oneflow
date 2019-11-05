#ifndef ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"

#ifdef WITH_CUDA

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <nccl.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

namespace oneflow {

template<typename T>
void CudaCheck(T error);

// CUDA: grid stride looping
#define CUDA_1D_KERNEL_LOOP(i, n)                                                                 \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

#define CUDA_1D_KERNEL_LOOP_T(type, i, n)                                                      \
  for (type i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

const int32_t kCudaThreadsNumPerBlock = 1024;
const int32_t kCudaMaxBlocksNum = 4096;

int32_t GetSMCudaMaxBlocksNum();
void InitGlobalCudaDeviceProp();
bool IsCuda9OnTuringDevice();

inline int32_t BlocksNum4ThreadsNum(const int32_t n) {
  return std::min((n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock, kCudaMaxBlocksNum);
}

inline int32_t SMBlocksNum4ThreadsNum(const int32_t n) {
  return std::min((n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock,
                  GetSMCudaMaxBlocksNum());
}

#define RUN_CUDA_KERNEL(func, device_ctx_ptr, thread_num, ...)           \
  func<<<SMBlocksNum4ThreadsNum(thread_num), kCudaThreadsNumPerBlock, 0, \
         (device_ctx_ptr)->cuda_stream()>>>(__VA_ARGS__)

size_t GetAvailableGpuMemSize(int dev_id);

#define CUDA_WORK_TYPE_SEQ          \
  OF_PP_MAKE_TUPLE_SEQ(kCompute)    \
  OF_PP_MAKE_TUPLE_SEQ(kCopyH2D)    \
  OF_PP_MAKE_TUPLE_SEQ(kCopyD2H)    \
  OF_PP_MAKE_TUPLE_SEQ(kNccl)       \
  OF_PP_MAKE_TUPLE_SEQ(kMix)        \
  OF_PP_MAKE_TUPLE_SEQ(kReduceCtrl) \
  OF_PP_MAKE_TUPLE_SEQ(kMdUpdt)

enum class CudaWorkType {
#define DECLARE_CUDA_WORK_TYPE(type) type,
  OF_PP_FOR_EACH_TUPLE(DECLARE_CUDA_WORK_TYPE, CUDA_WORK_TYPE_SEQ)
};

inline size_t GetCudaWorkTypeSize() { return OF_PP_SEQ_SIZE(CUDA_WORK_TYPE_SEQ); }

void NumaAwareCudaMallocHost(int32_t dev, void** ptr, size_t size);

template<typename T>
void NumaAwareCudaMallocHost(int32_t dev, T** ptr, size_t size) {
  NumaAwareCudaMallocHost(dev, reinterpret_cast<void**>(ptr), size);
}

#define CUDA_DATA_TYPE_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(float, CUDA_R_32F)  \
  OF_PP_MAKE_TUPLE_SEQ(double, CUDA_R_64F) \
  OF_PP_MAKE_TUPLE_SEQ(float16, CUDA_R_16F)

cudaDataType_t GetCudaDataType(DataType);

template<typename T>
struct CudaDataType;

#define SPECIALIZE_CUDA_DATA_TYPE(type_cpp, type_cuda) \
  template<>                                           \
  struct CudaDataType<type_cpp> : std::integral_constant<cudaDataType_t, type_cuda> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_CUDA_DATA_TYPE, CUDA_DATA_TYPE_SEQ);
#undef SPECIALIZE_CUDA_DATA_TYPE

class CudaCurrentDeviceGuard final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaCurrentDeviceGuard)
  explicit CudaCurrentDeviceGuard(int32_t dev_id);
  ~CudaCurrentDeviceGuard();

 private:
  int32_t saved_dev_id_ = -1;
};

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
