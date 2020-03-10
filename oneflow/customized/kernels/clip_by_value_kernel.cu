#include "oneflow/customized/kernels/clip_by_value_kernel.h"
#include "oneflow/core/kernel/util/cuda_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void CudaClipByMin(int64_t num_values, const T* values, const T* min_value, T* out_ptr) {
  ClipValuesByMin<DeviceType::kGPU>(num_values, values, *min_value, out_ptr);
}

template<typename T>
__global__ void CudaClipByMax(int64_t num_values, const T* values, const T* max_value, T* out_ptr) {
  ClipValuesByMax<DeviceType::kGPU>(num_values, values, *max_value, out_ptr);
}

template<typename T>
__global__ void CudaClipByMinMax(int64_t num_values, const T* values, const T* min_value,
                                 const T* max_value, T* out_ptr) {
  ClipValuesByMinMax<DeviceType::kGPU>(num_values, values, *min_value, *max_value, out_ptr);
}

}  // namespace

template<typename T>
struct ClipFunctor<DeviceType::kGPU, T> {
  __device__ __forceinline__ static T Min(const T value, const T min_value) {
    return min(value, min_value);
  }
  __device__ __forceinline__ static T Max(const T value, const T max_value) {
    return max(value, max_value);
  }
};

template<typename T>
struct ClipValuesUtil<DeviceType::kGPU, T> {
  static void ByMin(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                    T* out_ptr) {
    RUN_CUDA_KERNEL((CudaClipByMin<T>), ctx, num_values, num_values, values, min_value, out_ptr);
  }

  static void ByMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* max_value,
                    T* out_ptr) {
    RUN_CUDA_KERNEL((CudaClipByMax<T>), ctx, num_values, num_values, values, max_value, out_ptr);
  }

  static void ByMinMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                       const T* max_value, T* out_ptr) {
    RUN_CUDA_KERNEL((CudaClipByMinMax<T>), ctx, num_values, num_values, values, min_value,
                    max_value, out_ptr);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CLIP_VALUES_UTIL, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CLIP_BY_VALUE_KERNEL, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
