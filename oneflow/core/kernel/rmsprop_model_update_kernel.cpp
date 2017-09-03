#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RMSPropMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diffs_blob = BnInOp2Blob("model_diffs");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* mean_square_blob = BnInOp2Blob("mean_square");
  const RMSPropModelUpdateOpConf& conf = op()->op_conf().rmsprop_mdupdt_conf();
  const float batch_size = JobDesc::Singleton()->batch_size();
  const float learning_rate = conf.learning_rate();
  const float epsilon = conf.epsilon();
  float decay_rate = conf.decay_rate();
  if (*reinterpret_cast<int64_t*>(ctx.other) == 1) { decay_rate = 0.0f; }

  RMSPropMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(),
      static_cast<T>((1.0f - decay_rate) / (batch_size * batch_size)),
      static_cast<T>(learning_rate / batch_size), static_cast<T>(decay_rate),
      static_cast<T>(epsilon), model_blob->mut_dptr<T>(),
      mean_square_blob->mut_dptr<T>(), model_diffs_blob->dptr<T>());
}

template<DeviceType device_type, typename T>
void RMSPropMdUpdateKernel<device_type, T>::InitDataTmpBlobs(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FillConf mean_sqaure_fill_conf;
  mean_sqaure_fill_conf.mutable_constant_conf()->set_value(0.0f);
  KernelUtil<device_type, T>::Fill(ctx.device_ctx, mean_sqaure_fill_conf, 0,
                                   BnInOp2Blob("mean_square"));
}

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(const KernelCtx& ctx, const int64_t n, const T alpha,
                          const T learning_rate, const T decay_rate,
                          const T epsilon, T* model, T* mean_square,
                          const T* model_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        mean_square[i] =
            alpha * model_diff[i] * model_diff[i] + decay_rate * mean_square[i];
        model[i] -= learning_rate * model_diff[i]
                    / (std::sqrt(mean_square[i] + epsilon));
      }
    });
  }
};

}  // namespace oneflow
