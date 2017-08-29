#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MultinomialLogisticLossKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* prediction = BnInOp2BlobPtr("prediction");
  const Blob* label = BnInOp2BlobPtr("label");
  Blob* loss = BnInOp2BlobPtr("loss");
  Blob* loss_buff = BnInOp2BlobPtr("loss_buffer");

  MultinomialLogisticLossKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, prediction->shape().At(0), prediction->shape().At(1),
      prediction->dptr<T>(), label->dptr<int32_t>(), loss->mut_dptr<T>(),
      loss_buff->mut_dptr<T>());

  Blob* prediction_diff = BnInOp2BlobPtr(GenDiffBn("prediction"));
  if (prediction_diff != nullptr) {
    Memset<device_type>(ctx.device_ctx, prediction_diff->mut_dptr<T>(), 0,
                        prediction_diff->TotalByteSize());
    MultinomialLogisticLossKernelUtil<device_type, T>::Backward(
        ctx.device_ctx, prediction->shape().At(0), prediction->shape().At(1),
        prediction->dptr<T>(), label->dptr<int32_t>(),
        prediction_diff->mut_dptr<T>());
  }
}

template<typename T>
class MultinomialLogisticLossKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(DeviceCtx* ctx, const int64_t instance_num,
                      const int64_t num_of_classes, const T* prediction,
                      const int32_t* labels, T* loss, T* loss_buff) {
    ctx->cpu_stream()->SendWork([=]() {
      loss[0] = 0;
      for (int64_t i = 0; i < instance_num; ++i) {
        T prob =
            prediction[i * num_of_classes + static_cast<int64_t>(labels[i])];
        loss[0] -= SAFE_LOG(prob);
      }
    });
  }

  static void Backward(DeviceCtx* ctx, const int64_t instance_num,
                       const int64_t num_of_classes, const T* prediction,
                       const int32_t* labels, T* prediction_diff) {
    ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < instance_num; ++i) {
        int64_t label = static_cast<int64_t>(labels[i]);
        T prob = MAX_WITH_LOG_THRESHOLD(prediction[i * num_of_classes + label]);
        prediction_diff[i * num_of_classes + label] = -1 / prob;
      }
    });
  }
};  // namespace oneflow

namespace {

template<DeviceType device_type>
Kernel* CreateMultinomialLogisticLossKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> data_type2creator = {
#define MULTINOMIAL_LOGISTIC_LOSS_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto,                                                       \
   []() { return new MultinomialLogisticLossKernel<device_type, type_cpp>; }},
      FOR_EACH_PAIR(MULTINOMIAL_LOGISTIC_LOSS_KERNEL_ENTRY,
                    FLOATING_DATA_TYPE_SEQ)};
  return data_type2creator.at(
      op_conf.multinomial_logistic_loss_conf().prediction().data_type())();
}

}  // namespace

REGISTER_TEMPLATE_KERNEL_CREATOR(OperatorConf::kMultinomialLogisticLossConf,
                                 CreateMultinomialLogisticLossKernel);

}  // namespace oneflow
