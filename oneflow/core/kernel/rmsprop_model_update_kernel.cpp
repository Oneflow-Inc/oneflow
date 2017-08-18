#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void RMSPropMdUpdateKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* model_blob = BnInOp2BlobPtr("model");
  const Blob* model_diffs_blob = BnInOp2BlobPtr("model_diffs");
  Blob* mean_square_blob = BnInOp2BlobPtr("mean_square");
  const float batch_size = JobDesc::Singleton()->batch_size();
  const float learning_rate =
      op()->op_conf().rmsprop_mdupdt_conf().learning_rate();
  const float epsilon = op()->op_conf().rmsprop_mdupdt_conf().epsilon();
  float decay_rate = op()->op_conf().rmsprop_mdupdt_conf().decay_rate();
  if (*reinterpret_cast<int64_t*>(ctx.other) == 1) { decay_rate = 0.0f; }

  RMSPropMdUpdateKernelUtil<device_type, FloatingPointType>::UpdateMeanSquare(
      ctx, mean_square_blob->shape().elem_cnt(),
      static_cast<FloatingPointType>((1 - decay_rate)
                                     / std::pow(batch_size, 2)),
      static_cast<FloatingPointType>(decay_rate),
      mean_square_blob->mut_dptr<FloatingPointType>(),
      model_diffs_blob->dptr<FloatingPointType>());

  RMSPropMdUpdateKernelUtil<device_type, FloatingPointType>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(),
      model_blob->mut_dptr<FloatingPointType>(),
      model_diffs_blob->dptr<FloatingPointType>(),
      mean_square_blob->dptr<FloatingPointType>(),
      static_cast<FloatingPointType>(epsilon),
      static_cast<FloatingPointType>(learning_rate / batch_size));
}

template<DeviceType device_type, typename FloatingPointType>
void RMSPropMdUpdateKernel<device_type, FloatingPointType>::InitDataTmpBlobs(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FillConf mean_sqaure_fill_conf;
  mean_sqaure_fill_conf.mutable_constant_conf()->set_value(0.0f);
  KernelUtil<device_type, FloatingPointType>::Fill(
      ctx, mean_sqaure_fill_conf, 0, BnInOp2Blob("mean_square"));
}

template<typename FloatingPointType>
class RMSPropMdUpdateKernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropMdUpdateKernelUtil);
  RMSPropMdUpdateKernelUtil() = delete;

  static void UpdateMeanSquare(const KernelCtx& ctx, const int64_t n,
                               const FloatingPointType alpha,
                               const FloatingPointType decay_rate,
                               FloatingPointType* mean_square,
                               const FloatingPointType* model_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        mean_square[i] =
            alpha * model_diff[i] * model_diff[i] + decay_rate * mean_square[i];
      }
    });
  }

  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          FloatingPointType* model,
                          const FloatingPointType* model_diff,
                          const FloatingPointType* mean_square,
                          const FloatingPointType epsilon,
                          const FloatingPointType alpha) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        model[i] -=
            alpha * model_diff[i] / (std::sqrt(mean_square[i] + epsilon));
      }
    });
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(RMSPropMdUpdateKernelUtil);
INSTANTIATE_KERNEL_CLASS(RMSPropMdUpdateKernel);
REGISTER_KERNEL(OperatorConf::kRmspropMdupdtConf, RMSPropMdUpdateKernel);

}  // namespace oneflow
