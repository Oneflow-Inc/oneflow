#include "oneflow/core/kernel/accuracy_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void AccuracyKernel<device_type, PredType, LabelType>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  const Blob* label_blob = BnInOp2Blob("label");
  CHECK_EQ(label_blob->shape().NumAxes(), 1);
  Blob* accuracy_blob = BnInOp2Blob("accuracy");
  const int32_t top_k = this->kernel_conf().op_attribute().op_conf().accuracy_conf().top_k();
  int32_t n = prediction_blob->shape().At(0);
  int32_t d = prediction_blob->shape().Count(1);

  Memset<device_type>(ctx.device_ctx, accuracy_blob->mut_dptr<PredType>(), 0,
                      accuracy_blob->ByteSizeOfDataContentField());
  AccuracyKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, n, d, top_k, prediction_blob->dptr<PredType>(), label_blob->dptr<LabelType>(),
      accuracy_blob->mut_dptr<PredType>());
}

template<typename PredType, typename LabelType>
struct AccuracyKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int32_t n, const int32_t d, int32_t top_k,
                      const PredType* prediction, const LabelType* label, PredType* accuracy) {
    for (int32_t i = 0; i < n; ++i) {
      const LabelType label_i = label[i];
      const PredType pred_i = prediction[i * d + label_i];
      int32_t cnt = 1;
      for (int32_t j = 0; j < d; ++j) {
        if (prediction[i * d + j] > pred_i) {
          if (++cnt > top_k) { break; }
        }
      }
      if (cnt <= top_k) { *accuracy += 1; }
    }
    CHECK_LE(*accuracy, n);
  }
};

namespace {

Kernel* CreateAccuracyKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define ACCURACY_KERNEL_ENTRY(device_type, pred_type_pair, label_type_pair)                        \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new AccuracyKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair),                      \
                               OF_PP_PAIR_FIRST(label_type_pair)>();                               \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(ACCURACY_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                                kernel_conf.accuracy_conf().prediction_type(),
                                kernel_conf.accuracy_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kAccuracyConf, CreateAccuracyKernel);

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                      \
  template struct AccuracyKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                     OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
