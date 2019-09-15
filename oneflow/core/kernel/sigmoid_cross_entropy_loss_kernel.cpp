#include "oneflow/core/kernel/sigmoid_cross_entropy_loss_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SigmoidCrossEntropyLossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const SigmoidCrossEntropyLossOpConf& conf = this->op_conf().sigmoid_cross_entropy_loss_conf();
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* elementwise_loss = BnInOp2Blob("elementwise_loss");
  Blob* sum_buf = BnInOp2Blob("sum_buf");
  Blob* count = BnInOp2Blob("count");
  Blob* label_num = BnInOp2Blob("label_num");
  Blob* loss = BnInOp2Blob("loss");
  const int64_t n = prediction->shape().elem_cnt();
  const int64_t instance_num = prediction->shape().At(0);
  const Shape& prediction_shape = {instance_num, prediction->shape().Count(1)};
  const Shape& loss_shape = {instance_num, 1};
  SigmoidCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, conf, n, prediction->dptr<PredType>(), label->dptr<LabelType>(),
      elementwise_loss->mut_dptr<PredType>(), count->mut_dptr<PredType>());
  NdarrayUtil<device_type, PredType>::ReduceSum(
      ctx.device_ctx, XpuVarNdarray<PredType>(loss_shape, loss->mut_dptr<PredType>()),
      XpuVarNdarray<const PredType>(prediction_shape, elementwise_loss->dptr<PredType>()),
      XpuVarNdarray<PredType>(prediction_shape, sum_buf->mut_dptr<PredType>()));
  if (conf.normalize()) {
    NdarrayUtil<device_type, PredType>::ReduceSum(
        ctx.device_ctx, XpuVarNdarray<PredType>(loss_shape, label_num->mut_dptr<PredType>()),
        XpuVarNdarray<const PredType>(prediction_shape, count->dptr<PredType>()),
        XpuVarNdarray<PredType>(prediction_shape, sum_buf->mut_dptr<PredType>()));
    SigmoidCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::ClipByEpsilon(
        ctx.device_ctx, instance_num, label_num->mut_dptr<PredType>());
    KernelUtil<device_type, PredType>::Div(ctx.device_ctx, instance_num, loss->dptr<PredType>(),
                                           label_num->dptr<PredType>(), loss->mut_dptr<PredType>());
  }
  KernelUtil<device_type, PredType>::Scal(ctx.device_ctx, instance_num,
                                          static_cast<PredType>(conf.scale()),
                                          loss->mut_dptr<PredType>(), 1);
}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf&
SigmoidCrossEntropyLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.sigmoid_cross_entropy_loss_conf().loss_conf();
}

template<typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                      const PredType* prediction, const LabelType* label,
                      PredType* elementwise_loss, PredType* count) {
    FOR_RANGE(int64_t, index, 0, n) {
      if (label[index] == -1) {
        elementwise_loss[index] = 0.f;
        count[index] = 0.f;
      } else {
        elementwise_loss[index] =
            -1 * prediction[index] * (label[index] - (prediction[index] >= 0))
            + logf(1 + expf(prediction[index] - 2 * prediction[index] * (prediction[index] >= 0)));
        count[index] = 1.f;
      }
    }
  }

  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                       const PredType* prediction, const LabelType* label, PredType* pred_diff) {
    FOR_RANGE(int64_t, index, 0, n) {
      if (label[index] == -1) {
        pred_diff[index] = 0.f;
      } else {
        pred_diff[index] = 1.f / (1.f + expf(-prediction[index])) - label[index];
      }
    }
  }

  static void ClipByEpsilon(DeviceCtx* ctx, const int64_t n, PredType* x) {
    const PredType floor_val = 1e-5;
    FOR_RANGE(int64_t, index, 0, n) { x[index] = std::max(x[index], floor_val); }
  }
};

namespace {

Kernel* CreateSigmoidCrossEntropyLossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_ENTRY(device_type, pred_type_pair, label_type_pair)      \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new SigmoidCrossEntropyLossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair),       \
                                              OF_PP_PAIR_FIRST(label_type_pair)>();                \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SIGMOID_CROSS_ENTROPY_LOSS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                 kernel_conf.sigmoid_cross_entropy_loss_conf().loss_conf().prediction_type(),
                 kernel_conf.sigmoid_cross_entropy_loss_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kSigmoidCrossEntropyLossConf,
                        CreateSigmoidCrossEntropyLossKernel);

}  // namespace oneflow
