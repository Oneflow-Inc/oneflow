#include "oneflow/core/kernel/smooth_l1_loss_kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void SmoothL1LossForward(const int64_t N, const int64_t D,
                                                          const PredType* predict,
                                                          const LabelType* target,
                                                          const int64_t* inside_weights,
                                                          const int64_t* outside_weights,
                                                          PredType* loss) {
  TODO();
}

template<typename PredType, typename LabelType>
__global__ void SmoothL1LossBackward(const int64_t N, const int64_t D,
                                                         const PredType* predict,
                                                         const LabelType* target,
                                                         const int64_t* inside_weights,
                                                         const int64_t* outside_weights,
                                                         PredType* in_diff) {
  TODO();
}

}  // namespace

template<typename PredType, typename LabelType>
struct SmoothL1LossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t N, const int64_t D, const PredType* predict, const LabelType* target, const int64_t* inside_weights, const int64_t* outside_weights,
    PredType* loss) {
    SmoothL1LossForward<PredType>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(N, D, predict, target, loss);
  }
  static void Backward(DeviceCtx* ctx, const int64_t n, const int64_t w, const PredType* predict ,const LabelType* target, const int64_t* inside_weights, const int64_t* outside_weights,
                          PredType* in_diff) {
    SmoothL1LossBackward<PredType>
    <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(N, D, predict, target, in_diff);
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)        \
  template struct SmoothL1LossKernelUtil< \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
}  // namespace oneflow
