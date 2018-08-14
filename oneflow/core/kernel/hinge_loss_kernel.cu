#include "oneflow/core/kernel/hinge_loss_kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void HingeLossForwardTransSignGpu(const int64_t data_num, const int64_t pre_dim,
                                             const LabelType* label, PredType* tmp_diff) {
  CUDA_1D_KERNEL_LOOP(i, data_num) { tmp_diff[i * pre_dim + static_cast<int64_t>(label[i])] *= -1; }
}

template<typename PredType>
__global__ void HingeLossForwardMaxGpu(const int64_t data_num, const int64_t pre_dim,
                                       PredType* tmp_diff) {
  CUDA_1D_KERNEL_LOOP(i, data_num * pre_dim) {
    tmp_diff[i] = 1 + tmp_diff[i] > 0 ? 1 + tmp_diff[i] : 0;
  }
}

template<typename PredType>
__global__ void HingeLossBackwardTransSignGpu(const int64_t data_num, const int64_t pre_dim,
                                              const PredType* tmp_diff, PredType* pred_diff) {
  CUDA_1D_KERNEL_LOOP(i, data_num * pre_dim) { pred_diff[i] = (tmp_diff[i] > 0); }
}

template<typename PredType, typename LabelType>
__global__ void HingeLossBackwardL1Gpu(const int64_t data_num, const int64_t pre_dim,
                                       const LabelType* label, PredType* pred_diff) {
  CUDA_1D_KERNEL_LOOP(i, data_num) {
    pred_diff[i * pre_dim + static_cast<int64_t>(label[i])] *= -1;
  }
}

template<typename PredType>
__global__ void HingeLossBackwardL2Gpu(const int64_t data_num, const int64_t pre_dim,
                                       const PredType* tmp_diff, PredType* pred_diff) {
  CUDA_1D_KERNEL_LOOP(i, data_num * pre_dim) { pred_diff[i] = 2 * tmp_diff[i] * pred_diff[i]; }
}

}  // namespace

template<typename PredType, typename LabelType>
struct HingeLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t data_num, const int64_t pre_dim,
                      const PredType* pred, const LabelType* label, const OperatorConf op_conf,
                      PredType* tmp_diff, PredType* loss) {
    HingeLossForwardTransSignGpu<<<BlocksNum4ThreadsNum(data_num), kCudaThreadsNumPerBlock, 0,
                                   ctx->cuda_stream()>>>(data_num, pre_dim, label, tmp_diff);
    HingeLossForwardMaxGpu<<<BlocksNum4ThreadsNum(data_num * pre_dim), kCudaThreadsNumPerBlock, 0,
                             ctx->cuda_stream()>>>(data_num, pre_dim, tmp_diff);
    switch (op_conf.hinge_loss_conf().norm()) {
      case L1:
        for (int64_t i = 0; i < data_num; ++i) {
          PredType tmp_storage[pre_dim];
          KernelUtil<DeviceType::kGPU, PredType>::Sum(ctx, pre_dim, tmp_diff + i * pre_dim,
                                                      loss + i, tmp_storage,
                                                      pre_dim * sizeof(PredType));
        }
        break;
      case L2:
        for (int64_t i = 0; i < data_num; ++i) {
          KernelUtil<DeviceType::kGPU, PredType>::Dot(ctx, pre_dim, tmp_diff + i * pre_dim, 1,
                                                      tmp_diff + i * pre_dim, 1, loss + i);
        }
        break;
      default: LOG(FATAL) << "Invalid norm method in " << op_conf.name();
    }
  }

  static void Backward(DeviceCtx* ctx, const int64_t data_num, const int64_t pre_dim,
                       const PredType* tmp_diff, const LabelType* label, const OperatorConf op_conf,
                       PredType* pred_diff) {
    HingeLossBackwardTransSignGpu<<<BlocksNum4ThreadsNum(data_num * pre_dim),
                                    kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        data_num, pre_dim, tmp_diff, pred_diff);
    HingeLossBackwardL1Gpu<<<BlocksNum4ThreadsNum(data_num), kCudaThreadsNumPerBlock, 0,
                             ctx->cuda_stream()>>>(data_num, pre_dim, label, pred_diff);
    switch (op_conf.hinge_loss_conf().norm()) {
      case L1: break;
      case L2:
        HingeLossBackwardL2Gpu<<<BlocksNum4ThreadsNum(data_num * pre_dim), kCudaThreadsNumPerBlock,
                                 0, ctx->cuda_stream()>>>(data_num, pre_dim, tmp_diff, pred_diff);
        break;
      default: LOG(FATAL) << "Invalid norm method in " << op_conf.name();
    }
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                       \
  template struct HingeLossKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                      OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
}  // namespace oneflow
