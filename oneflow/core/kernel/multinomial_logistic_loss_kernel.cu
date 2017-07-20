#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
__global__ void MultinomialLogisticLossForwardGpu(
    const int64_t piece_size, const int64_t num_of_classes,
    const FloatingPointType* prediction, const FloatingPointType* labels,
    FloatingPointType* loss) {
  // 1 block and 1 thread
  for (int64_t i = 0; i < piece_size; ++i) {
    int64_t label = labels[i];
    // FloatingPointType prob = max(prediction[i * num_of_classes + label],
    //                             FloatingPointType(kLOG_THRESHOLD));
    FloatingPointType prob = prediction[i * num_of_classes + label];
    prob = prob > FloatingPointType(kLOG_THRESHOLD)
               ? prob
               : FloatingPointType(kLOG_THRESHOLD);
    loss[0] -= logf(prob);
  }
  loss[0] = loss[0] / piece_size;
}

template<typename FloatingPointType>
__global__ void MultinomialLogisticLossBackwardGpu(
    const int64_t piece_size, const int64_t num_of_classes,
    const FloatingPointType* prediction, const FloatingPointType* labels,
    FloatingPointType* prediction_diff) {
  const FloatingPointType scale = -1.0 / piece_size;
  // piece_size = nthreads
  CUDA_1D_KERNEL_LOOP(i, piece_size) {
    int64_t label = labels[i];
    FloatingPointType prob = prediction[i * num_of_classes + label];
    prob = prob > FloatingPointType(kLOG_THRESHOLD)
               ? prob
               : FloatingPointType(kLOG_THRESHOLD);
    prediction_diff[i * num_of_classes + label] = scale / prob;
  }
}

}  // namespace

template<typename FloatingPointType>
class MultinomialLogisticLossKernelUtil<DeviceType::kGPU, FloatingPointType>
    final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t piece_size,
                      const int64_t num_of_classes,
                      const FloatingPointType* prediction,
                      const FloatingPointType* labels,
                      FloatingPointType* loss) {
    MultinomialLogisticLossForwardGpu<FloatingPointType>
        <<<1, 1, 0,  // 1 block and 1 thread
           ctx.device_ctx->cuda_stream()>>>(piece_size, num_of_classes,
                                            prediction, labels, loss);
    // MultinomialLogisticLossForwardGpu<FloatingPointType>
    //     <<<BlocksNum4ThreadsNum(piece_size), kCudaThreadsNumPerBlock, 0,
    //        ctx.device_ctx->cuda_stream()>>>(piece_size, num_of_classes,
    //        prediction, labels, loss);
  }

  static void Backward(const KernelCtx& ctx, const int64_t piece_size,
                       const int64_t num_of_classes,
                       const FloatingPointType* prediction,
                       const FloatingPointType* labels,
                       FloatingPointType* prediction_diff) {
    MultinomialLogisticLossBackwardGpu<FloatingPointType>
        <<<BlocksNum4ThreadsNum(piece_size), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            piece_size, num_of_classes, prediction, labels, prediction_diff);
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(MultinomialLogisticLossKernelUtil);

}  // namespace oneflow
