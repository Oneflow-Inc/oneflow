#include "oneflow/core/kernel/smooth_l1_kernel.h"
#include <cub/cub.cuh>
#include <math.h>

namespace oneflow {

namespace {

template<typename T>
__global__ void SmoothL1Forward(const int64_t elem_cnt,
                                    const T* prediction, const T* label, const T* inside_weights,
                                    const T* outside_weights, const float beta, const float scale,
                                    T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    T x = inside_weights[i] * (prediction[i] - label[i]);
    T abs_x = std::abs(x);
    if (abs_x < beta) {
      out[i] = 0.5 * x * x / beta;
    } else {
      out[i] = abs_x - 0.5 * beta;
    }
    out[i] *= scale * outside_weights[i];
  }
}
      
template<typename T>
__global__ void SmoothL1Backward(const int64_t elem_cnt,
                                      const T* prediction, const T* label, const T* inside_weights,
                                      const T* outside_weights, const float beta, const float scale,
                                      T* prediction_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    T x = inside_weights[i] * (prediction[i] - label[i]);
    T abs_x = std::abs(x);
    if (abs_x < beta) {
      prediction_diff[i] = x / beta;
    } else {
      prediction_diff[i] = (x > ZeroVal<T>::value) - (x < ZeroVal<T>::value);
    }
    prediction_diff[i] *= scale * inside_weights[i] * outside_weights[i];
  }
}

}  // namespace

template<typename T>
struct SmoothL1KernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt,
                      const T* prediction, const T* label, const T* inside_weights,
                      const T* outside_weights, const float beta, const float scale, T* out) {
    SmoothL1Forward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(elem_cnt, prediction, label, inside_weights,
                                 outside_weights, beta, scale, out);
  }
  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt,
                       const T* prediction, const T* label, const T* inside_weights,
                       const T* outside_weights, const float beta, const float scale, T* prediction_diff) {
    SmoothL1Backward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(elem_cnt, prediction, label, inside_weights,
                                 outside_weights, beta, scale, prediction_diff);
  }
};

#define INSTANTIATE_SMOOTH_L1_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SmoothL1KernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SMOOTH_L1_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_SMOOTH_L1_KERNEL_UTIL

}  // namespace oneflow
