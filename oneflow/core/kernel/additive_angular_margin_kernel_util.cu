#include "oneflow/core/kernel/additive_angular_margin_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t GetOffset(const int64_t batch_idx, const int64_t labels_num,
                             const int64_t lower_bound, const K* label) {
  const int64_t idx = label[batch_idx] - lower_bound;
  if (idx >= 0 && idx < labels_num) {
    return batch_idx * labels_num + idx;
  } else {
    return -1;
  }
}

template<typename T, typename K>
__global__ void AdditiveAngularMarginForwardGpu(const int64_t batch_num, const int64_t labels_num,
                                                const int64_t lower_bound, const T cos_m,
                                                const T sin_m, const T* in, const K* label,
                                                T* sin_theta_data, T* out) {
  CUDA_1D_KERNEL_LOOP(i, batch_num) {
    const int64_t idx = GetOffset<K>(i, labels_num, lower_bound, label);
    if (idx != -1) {
      sin_theta_data[i] = sqrt(1 - in[idx] * in[idx]);
      out[idx] = in[idx] * cos_m - sin_theta_data[i] * sin_m;
      sin_theta_data[i] = in[idx] / sin_theta_data[i];
    }
  }
}

template<typename T, typename K>
__global__ void AdditiveAngularMarginBackwardGpu(const int64_t batch_num, const int64_t labels_num,
                                                 const int64_t lower_bound, const T cos_m,
                                                 const T sin_m, const T* out_diff, const K* label,
                                                 const T* sin_theta_data, T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, batch_num) {
    const int64_t idx = GetOffset<K>(i, labels_num, lower_bound, label);
    if (idx != -1) { in_diff[idx] = in_diff[idx] * (1 * cos_m + sin_theta_data[i] * sin_m); }
  }
}

}  // namespace

template<typename T, typename K>
struct AdditiveAngularMarginKernelUtilImpl<DeviceType::kGPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                      const int64_t lower_bound, const T cos_m, const T sin_m, const T* in,
                      const K* label, T* sin_theta_data, T* out);
  static void Backward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                       const int64_t lower_bound, const T cos_m, const T sin_m, const T* out_diff,
                       const K* label, const T* sin_theta_data, T* in_diff);
};

template<typename T, typename K>
void AdditiveAngularMarginKernelUtilImpl<DeviceType::kGPU, T, K>::Forward(
    DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num, const int64_t lower_bound,
    const T cos_m, const T sin_m, const T* in, const K* label, T* sin_theta_data, T* out) {
  AdditiveAngularMarginForwardGpu<T, K>
      <<<BlocksNum4ThreadsNum(batch_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          batch_num, labels_num, lower_bound, cos_m, sin_m, in, label, sin_theta_data, out);
}

template<typename T, typename K>
void AdditiveAngularMarginKernelUtilImpl<DeviceType::kGPU, T, K>::Backward(
    DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num, const int64_t lower_bound,
    const T cos_m, const T sin_m, const T* out_diff, const K* label, const T* sin_theta_data,
    T* in_diff) {
  AdditiveAngularMarginBackwardGpu<T, K>
      <<<BlocksNum4ThreadsNum(batch_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          batch_num, labels_num, lower_bound, cos_m, sin_m, out_diff, label, sin_theta_data,
          in_diff);
}

#define INSTANTIATE_ADDITIVE_ANGULAR_MARGIN_KERNEL_UTIL_IMPL_GPU(in_type_pair, index_type_pair) \
  template struct AdditiveAngularMarginKernelUtilImpl<                                          \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ADDITIVE_ANGULAR_MARGIN_KERNEL_UTIL_IMPL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_ADDITIVE_ANGULAR_MARGIN_KERNEL_UTIL_IMPL_GPU

}  // namespace oneflow
