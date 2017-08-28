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

  RMSPropMdUpdateKernelUtil<device_type, T>::UpdateMeanSquare(
      ctx, mean_square_blob->shape().elem_cnt(),
      static_cast<T>((1.0f - decay_rate) / (batch_size * batch_size)),
      static_cast<T>(decay_rate), mean_square_blob->mut_dptr<T>(),
      model_diffs_blob->dptr<T>());

  RMSPropMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), model_blob->mut_dptr<T>(),
      model_diffs_blob->dptr<T>(), mean_square_blob->dptr<T>(),
      static_cast<T>(epsilon), static_cast<T>(learning_rate / batch_size));
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
  static void UpdateMeanSquare(const KernelCtx& ctx, const int64_t n,
                               const T alpha, const T decay_rate,
                               T* mean_square, const T* model_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        mean_square[i] =
            alpha * model_diff[i] * model_diff[i] + decay_rate * mean_square[i];
      }
    });
  }

  static void UpdateModel(const KernelCtx& ctx, const int64_t n, T* model,
                          const T* model_diff, const T* mean_square,
                          const T epsilon, const T alpha) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        model[i] -=
            alpha * model_diff[i] / (std::sqrt(mean_square[i] + epsilon));
      }
    });
  }
};

namespace {

template<DeviceType device_type>
Kernel* CreateRmspropMdUpdateKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> data_type2creator = {
#define RMSPROP_MDUPDATE_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto,                                              \
   []() { return new RMSPropMdUpdateKernel<device_type, type_cpp>; }},
      FOR_EACH_PAIR(RMSPROP_MDUPDATE_KERNEL_ENTRY, FLOATING_DATA_TYPE_PAIR())};
  return data_type2creator.at(JobDesc::Singleton()->default_data_type())();
}

}  // namespace

REGISTER_TEMPLATE_KERNEL_CREATOR(OperatorConf::kRmspropMdupdtConf,
                                 CreateRmspropMdUpdateKernel);

}  // namespace oneflow
