#include "oneflow/core/kernel/salr_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void SALRMdUpdateKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* model_blob = BnInOp2BlobPtr("model");
  Blob* model_diff_blob = BnInOp2BlobPtr("model_diff");
  Blob* learning_rate_blob = BnInOp2BlobPtr("learning_rate");
  Blob* last_diff_flag_blob = BnInOp2BlobPtr("last_diff_flag");
  float epsilon = op()->op_conf().salr_mdupdt_conf().epsilon()
                  / JobDesc::Singleton()->batch_size();
  float delta = op()->op_conf().salr_mdupdt_conf().delta();

  SALRMdUpdateKernelUtil<device_type, FloatingPointType>::UpdateLearningRate(
      ctx, learning_rate_blob->shape().elem_cnt(),
      static_cast<FloatingPointType>(delta),
      static_cast<FloatingPointType*>(last_diff_flag_blob->mut_dptr()),
      static_cast<const FloatingPointType*>(model_diff_blob->dptr()),
      static_cast<FloatingPointType*>(learning_rate_blob->mut_dptr()));

  SALRMdUpdateKernelUtil<device_type, FloatingPointType>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(),
      static_cast<FloatingPointType*>(model_blob->mut_dptr()),
      static_cast<const FloatingPointType*>(model_diff_blob->dptr()),
      static_cast<const FloatingPointType*>(learning_rate_blob->mut_dptr()),
      static_cast<FloatingPointType>(epsilon));
}

template<typename FloatingPointType>
class SALRMdUpdateKernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SALRMdUpdateKernelUtil);
  SALRMdUpdateKernelUtil() = delete;

  // if diff(t) * diff(t-1) > 0
  // then learning_rate = learning_rate + delta
  // else learning_rate = learning_rate * (1 - delta)
  static void UpdateLearningRate(const KernelCtx& ctx, const int64_t n,
                                 const FloatingPointType delta,
                                 FloatingPointType* last_diff_flag,
                                 const FloatingPointType* model_diff,
                                 FloatingPointType* learning_rate) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        if (last_diff_flag[i] * model_diff[i] > 0) {
          learning_rate[i] = learning_rate[i] + delta;
        } else {
          learning_rate[i] = learning_rate[i] * (1 - delta);
        }
        last_diff_flag[i] = model_diff[i] > 0 ? 1 : -1;
      }
    });
  }

  // model -= (-epsilon) * learning_rate * model_diff
  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          FloatingPointType* model,
                          const FloatingPointType* model_diff,
                          const FloatingPointType* learning_rate,
                          const FloatingPointType epsilon) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        model[i] -= (-epsilon) * learning_rate[i] * model_diff[i];
      }
    });
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(SALRMdUpdateKernelUtil);
INSTANTIATE_KERNEL_CLASS(SALRMdUpdateKernel);
REGISTER_KERNEL(OperatorConf::kSalrMdupdtConf, SALRMdUpdateKernel);

}  // namespace oneflow
