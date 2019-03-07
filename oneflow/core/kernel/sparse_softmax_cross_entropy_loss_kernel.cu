#include "oneflow/core/kernel/sparse_softmax_cross_entropy_loss_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void SparseSoftmaxCrossEntropyLossBackwardSub(const int64_t n, const int64_t w,
                                                         const LabelType* label,
                                                         PredType* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) { in_diff[i * w + static_cast<int64_t>(label[i])] -= 1; }
}

template<typename LabelType>
__global__ void SoftmaxLossBackwardSubHalfGpu(const int64_t n, const int64_t w,
                                              const LabelType* label, half* in_diff) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t index = i * w + static_cast<int64_t>(label[i]);
    in_diff[index] = __hsub(in_diff[index], __float2half(1.0));
  }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

}  // namespace

template<typename PredType, typename LabelType>
struct SparseSoftmaxCrossEntropyLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w, const LabelType* label,
                          PredType* in_diff) {
    SparseSoftmaxCrossEntropyLossBackwardSub<PredType, LabelType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, w, label,
                                                                                      in_diff);
  }
};

template<typename LabelType>
struct SparseSoftmaxCrossEntropyLossKernelUtil<DeviceType::kGPU, float16, LabelType> {
  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w, const LabelType* label,
                          float16* in_diff) {
    SoftmaxLossBackwardSubHalfGpu<LabelType>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, w, label, reinterpret_cast<half*>(in_diff));
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)        \
  template struct SparseSoftmaxCrossEntropyLossKernelUtil< \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ)
}  // namespace oneflow
