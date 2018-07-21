#include "oneflow/core/kernel/smooth_l1_loss_kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void SmoothL1LossForward(const int64_t N, const int64_t D, const PredType* prediction,
                                    const LabelType* label, const int8_t* inside_weights,
                                    const int8_t* outside_weights, const float beta,
                                    const float scale, PredType* loss_buf) {
  int64_t elem_cnt = N * D;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    PredType x = inside_weights[i] * (prediction[i] - label[i]);
    PredType abs_x = abs(x);
    if (abs_x < beta) {
      loss_buf[i] = 0.5 * x * x / beta;
    } else {
      loss_buf[i] = abs_x - 0.5 * beta;
    }
    loss_buf[i] *= scale / elem_cnt * outside_weights[i];
  }
}

template<typename PredType, typename LabelType>
__global__ void SmoothL1LossBackward(const int64_t N, const int64_t D, const PredType* prediction,
                                     const LabelType* label, const int8_t* inside_weights,
                                     const int8_t* outside_weights, const float beta,
                                     const float scale, PredType* in_diff) {
  int64_t elem_cnt = N * D;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    PredType x = inside_weights[i] * (prediction[i] - label[i]);
    PredType abs_x = abs(x);
    if (abs_x < beta) {
      in_diff[i] = x / beta;
    } else {
      in_diff[i] = x > 0 ? 1 : -1;
    }
    in_diff[i] *= scale / elem_cnt * outside_weights[i];
  }
}

}  // namespace

template<typename PredType, typename LabelType>
struct SmoothL1LossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t N, const int64_t D, const PredType* prediction,
                      const LabelType* label, const int8_t* inside_weights,
                      const int8_t* outside_weights, const PredType* const_all_one,
                      const float beta, const float scale, PredType* loss_buf, PredType* loss) {
    SmoothL1LossForward<PredType>
        <<<BlocksNum4ThreadsNum(N * D), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            N, D, prediction, label, inside_weights, outside_weights, beta, scale, loss_buf);
    KernelUtil<DeviceType::kGPU, PredType>::Dot(ctx, N * D, loss_buf, 1, const_all_one, 1, loss);
  }
  static void Backward(DeviceCtx* ctx, const int64_t N, const int64_t D, const PredType* prediction,
                       const LabelType* label, const int8_t* inside_weights,
                       const int8_t* outside_weights, const float beta, const float scale,
                       PredType* in_diff) {
    SmoothL1LossBackward<PredType>
        <<<BlocksNum4ThreadsNum(N * D), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            N, D, prediction, label, inside_weights, outside_weights, beta, scale, in_diff);
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                          \
  template struct SmoothL1LossKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                         OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
}  // namespace oneflow
