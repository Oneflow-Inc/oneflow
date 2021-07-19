#ifndef ONEFLOW_CORE_DEVICE_ROCM_UTIL_H_
#define ONEFLOW_CORE_DEVICE_ROCM_UTIL_H_

#include "oneflow/core/common/data_type.h"

#ifdef WITH_ROCM

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <miopen/miopen.h>
#include <rccl.h>

namespace oneflow {

#define OF_ROCM_CHECK(condition)                                                               \
  for (hipError_t _of_hip_check_status = (condition); _of_hip_check_status != hipSuccess;) \
  LOG(FATAL) << "Check failed: " #condition " : " << hipGetErrorString(_of_hip_check_status) \
             << " (" << _of_hip_check_status << ") "

#define OF_HIPBLAS_CHECK(condition)                                                                 \
  for (hipblasStatus_t _of_hipblas_check_status = (condition);                                       \
       _of_hipblas_check_status != HIPBLAS_STATUS_SUCCESS;)                                          \
  LOG(FATAL) << "Check failed: " #condition " : " << " (" << _of_hipblas_check_status << ") "

#define OF_MIOPEN_CHECK(condition)                                                                \
  for (miopenStatus_t _of_cudnn_check_status = (condition);                                       \
       _of_cudnn_check_status != miopenStatusSuccess;)                                          \
  LOG(FATAL) << "Check failed: " #condition " : " << miopenGetErrorString(_of_cudnn_check_status) \
             << " (" << _of_cudnn_check_status << ") "

#define OF_NCCL_CHECK(condition)                                                                \
  for (ncclResult_t _of_nccl_check_status = (condition); _of_nccl_check_status != ncclSuccess;) \
  LOG(FATAL) << "Check failed: " #condition " : " << ncclGetErrorString(_of_nccl_check_status)  \
             << " (" << _of_nccl_check_status << ") "

template<typename T>
void RocmCheck(T error);

// ROCM: grid stride looping
#define ROCM_1D_KERNEL_LOOP(i, n)                                                                 \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

#define ROCM_1D_KERNEL_LOOP_T(type, i, n)                                                      \
  for (type i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

const int32_t kRocmThreadsNumPerBlock = 512;
const int32_t kRocmMaxBlocksNum = 8192;
const int32_t kRocmWarpSize = 32;

// 48KB, max byte size of shared memroy per thread block
// TODO: limit of shared memory should be different for different arch
const int32_t kRocmMaxSharedMemoryByteSize = 48 << 10;

int32_t GetSMRocmMaxBlocksNum();
void InitGlobalRocmDeviceProp();

inline int32_t BlocksNum4ThreadsNum(const int32_t n) {
  CHECK_GT(n, 0);
  return std::min((n + kRocmThreadsNumPerBlock - 1) / kRocmThreadsNumPerBlock, kRocmMaxBlocksNum);
}

inline int32_t SMBlocksNum4ThreadsNum(const int32_t n) {
  CHECK_GT(n, 0);
  return std::min((n + kRocmThreadsNumPerBlock - 1) / kRocmThreadsNumPerBlock,
                  GetSMRocmMaxBlocksNum());
}

#define RUN_ROCM_KERNEL(func, device_ctx_ptr, thread_num, shared_mem_bytes, ...) \
    hipLaunchKernelGGL(func, dim3(SMBlocksNum4ThreadsNum(thread_num)), dim3(kRocmThreadsNumPerBlock), \
        shared_mem_bytes, (device_ctx_ptr)->rocm_stream(), __VA_ARGS__);

size_t GetAvailableGpuMemSize(int dev_id);

#define ROCM_WORK_TYPE_SEQ       \
  OF_PP_MAKE_TUPLE_SEQ(kCompute) \
  OF_PP_MAKE_TUPLE_SEQ(kCopyH2D) \
  OF_PP_MAKE_TUPLE_SEQ(kCopyD2H) \
  OF_PP_MAKE_TUPLE_SEQ(kNccl)    \
  OF_PP_MAKE_TUPLE_SEQ(kMix)     \
  OF_PP_MAKE_TUPLE_SEQ(kDecodeH2D)

enum class CudaWorkType {
#define DECLARE_ROCM_WORK_TYPE(type) type,
  OF_PP_FOR_EACH_TUPLE(DECLARE_ROCM_WORK_TYPE, ROCM_WORK_TYPE_SEQ)
};

inline size_t GetCudaWorkTypeSize() { return OF_PP_SEQ_SIZE(ROCM_WORK_TYPE_SEQ); }

void NumaAwareRocmMallocHost(int32_t dev, void** ptr, size_t size);

template<typename T>
void NumaAwareRocmMallocHost(int32_t dev, T** ptr, size_t size) {
  NumaAwareRocmMallocHost(dev, reinterpret_cast<void**>(ptr), size);
}

// Set the CPU affinity to the closest processor(s) of a particular GPU.
void RocmDeviceSetCpuAffinity(int32_t dev);

#define ROCM_DATA_TYPE_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(float, HIPBLAS_R_32F)  \
  OF_PP_MAKE_TUPLE_SEQ(double, HIPBLAS_R_64F) \
  OF_PP_MAKE_TUPLE_SEQ(float16, HIPBLAS_R_16F)

hipblasDatatype_t GetRocmDataType(DataType);

template<typename T>
struct RocmDataType;

#define SPECIALIZE_ROCM_DATA_TYPE(type_cpp, type_rocm) \
  template<>                                           \
  struct RocmDataType<type_cpp> : std::integral_constant<hipblasDatatype_t, type_rocm> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_ROCM_DATA_TYPE, ROCM_DATA_TYPE_SEQ);
#undef SPECIALIZE_ROCM_DATA_TYPE

class RocmCurrentDeviceGuard final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RocmCurrentDeviceGuard)
  explicit RocmCurrentDeviceGuard(int32_t dev_id);
  RocmCurrentDeviceGuard();
  ~RocmCurrentDeviceGuard();

 private:
  int32_t saved_dev_id_ = -1;
};

}  // namespace oneflow

#endif  // WITH_ROCM

#endif  // ONEFLOW_CORE_DEVICE_ROCM_UTIL_H_
