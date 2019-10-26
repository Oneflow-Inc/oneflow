#include "oneflow/core/kernel/sparse_softmax_cross_entropy_grad_kernel.h"

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void SparseSoftmaxCrossEntropyGradBackwardSub(const int64_t n, const int64_t w,
                                                         const int64_t lower_bound, const T* dy,
                                                         const K* label, T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int64_t idx = label[i] - lower_bound;
    if (idx >= 0 && idx < w) { in_diff[i * w + idx] = dy[i] * (in_diff[i * w + idx] - 1); }
  }
}

}  // namespace

template<typename T, typename K>
struct SparseSoftmaxCrossEntropyGradKernelUtil<DeviceType::kGPU, T, K> {
  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const int64_t lower_bound, const T* dy, const K* label, T* in_diff) {
    SparseSoftmaxCrossEntropyGradBackwardSub<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, w, lower_bound, dy, label, in_diff);
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)        \
  template struct SparseSoftmaxCrossEntropyGradKernelUtil< \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
}  // namespace oneflow
