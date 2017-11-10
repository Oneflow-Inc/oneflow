#include "oneflow/core/kernel/lars_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LARSMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diffs_blob = BnInOp2Blob("model_diffs");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  Blob* temp_blob = BnInOp2Blob("temp");
  const LARSModelUpdateOpConf& conf = op()->op_conf().lars_mdupdt_conf();
  const int64_t batch_num = *reinterpret_cast<int64_t*>(ctx.other) - 1;
  const int64_t total_batch_num = JobDesc::Singleton()->total_batch_num();
  const float batch_size = JobDesc::Singleton()->batch_size();
  // t = batch_size
  // T = total_batch_size
  // learning_rate = base_learning_rate * (1 - t / T) ^ 2
  const float learning_rate =
      conf.learning_rate()
      * (1 - static_cast<float>(batch_num) / total_batch_num)
      * (1 - static_cast<float>(batch_num) / total_batch_num);
  const float lars_coefficient = conf.lars_coefficient();
  const float momentum = conf.momentum();
  const float weight_decay = conf.weight_decay();

  LARSMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), static_cast<T>(lars_coefficient),
      static_cast<T>(learning_rate / batch_size), static_cast<T>(momentum),
      static_cast<T>(weight_decay), model_blob->mut_dptr<T>(),
      momentum_blob->mut_dptr<T>(), temp_blob->mut_dptr<T>(),
      model_diffs_blob->dptr<T>());
}

template<DeviceType device_type, typename T>
void LARSMdUpdateKernel<device_type, T>::InitDataTmpBlobs(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FillConf momentum_fill_conf;
  momentum_fill_conf.mutable_constant_conf()->set_value(0.0f);
  KernelUtil<device_type, T>::Fill(ctx.device_ctx, momentum_fill_conf, 0,
                                   BnInOp2Blob("momentum"));
  FillConf temp_fill_conf;
  temp_fill_conf.mutable_constant_conf()->set_value(0.0f);
  KernelUtil<device_type, T>::Fill(ctx.device_ctx, temp_fill_conf, 0,
                                   BnInOp2Blob("temp"));
}

template<typename T>
class LARSMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          const T lars_coefficient, const T learning_rate,
                          const T m, const T weight_decay, T* model,
                          T* momentum, T* temp, const T* model_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      T model_norm = 0;
      T model_diff_norm = 0;
      for (int64_t i = 0; i != n; ++i) {
        model_norm += model[i] * model[i];
        model_diff_norm += model_diff[i] * model_diff[i];
      }
      const T local_lr = learning_rate * lars_coefficient * model_norm
                         / (model_diff_norm + weight_decay * model_norm);
      for (int64_t i = 0; i != n; ++i) {
        momentum[i] = m * momentum[i]
                      + local_lr * (model_diff[i] + weight_decay * model[i]);
        model[i] = model[i] - momentum[i];
      }
    });
  }
};

namespace {

Kernel* CreateLARSMdUpdateKernel(const OpContext& op_ctx) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define MODEL_UPDATE_KERNEL_ENTRY(device_type, data_type_pair)          \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() {   \
     return new LARSMdUpdateKernel<device_type,                         \
                                   OF_PP_PAIR_FIRST(data_type_pair)>(); \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
          MODEL_UPDATE_KERNEL_ENTRY, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(
      op_ctx.device_type(), op_ctx.bn_in_op2data_type().at("model_diffs")))();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kLarsMdupdtConf,
                         CreateLARSMdUpdateKernel))

}  // namespace oneflow
