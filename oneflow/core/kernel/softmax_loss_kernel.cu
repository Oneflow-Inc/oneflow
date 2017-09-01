#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/softmax_loss_kernel.h"

namespace oneflow {

namespace {

template<typename T, typename LabelType>
__global__ void SoftmaxLossForwardTmp(const int64_t n, const int64_t w,
                                      const LabelType* label, const T* prob,
                                      T* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    tmp[i] = -SAFE_LOG(prob[i * w + static_cast<int64_t>(label[i])]);
  }
}

template<typename T, typename LabelType>
__global__ void SoftmaxLossBackwardSub(const int64_t n, const int64_t w,
                                       const LabelType* label, T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    in_diff[i * w + static_cast<int64_t>(label[i])] -= 1;
  }
}

}  // namespace

template<typename T, typename LabelType>
class SoftmaxLossKernelUtil<DeviceType::kGPU, T, LabelType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxLossKernelUtil);
  SoftmaxLossKernelUtil() = delete;

  static void ComputeLoss(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const LabelType* label, const T* prob, T* tmp,
                          T* loss) {
    SoftmaxLossForwardTmp<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock,
                               0, ctx->cuda_stream()>>>(n, w, label, prob, tmp);
    KernelUtil<DeviceType::kGPU, T>::Sum(ctx, n, tmp, loss, tmp, sizeof(T) * n);
  }

  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const LabelType* label, T* in_diff) {
    SoftmaxLossBackwardSub<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(n, w, label, in_diff);
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)                      \
  template class SoftmaxLossKernelUtil<DeviceType::kGPU,                 \
                                       OF_PP_PAIR_FIRST(data_type_pair), \
                                       OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ)
}  // namespace oneflow
