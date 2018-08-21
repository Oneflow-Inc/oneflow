#include "oneflow/core/kernel/smooth_l1_loss_kernel.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SmoothL1LossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  const Blob* inside_weights = BnInOp2Blob("inside_weights");
  const Blob* outside_weights = BnInOp2Blob("outside_weights");
  Blob* loss = BnInOp2Blob("loss");
  Blob* pred_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  auto kernel_conf = this->kernel_conf();
  const float beta = kernel_conf.op_attribute().op_conf().smooth_l1_loss_conf().beta();
  const float scale = kernel_conf.op_attribute().op_conf().smooth_l1_loss_conf().scale();
  int64_t instance_num = BnInOp2Blob("prediction")->shape().At(0);
  int64_t instance_dim = BnInOp2Blob("prediction")->shape().Count(1);

  Memset<device_type>(ctx.device_ctx, loss->mut_dptr(), 0, loss->ByteSizeOfDataContentField());
  Memset<device_type>(ctx.device_ctx, pred_diff_blob->mut_dptr(), 0,
                      pred_diff_blob->ByteSizeOfDataContentField());

  SmoothL1LossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, instance_num, instance_dim, prediction->dptr<PredType>(),
      label->dptr<LabelType>(), inside_weights->dptr<PredType>(), outside_weights->dptr<PredType>(),
      beta, scale, loss->mut_dptr<PredType>());

  SmoothL1LossKernelUtil<device_type, PredType, LabelType>::Backward(
      ctx.device_ctx, instance_num, instance_dim, prediction->dptr<PredType>(),
      label->dptr<LabelType>(), inside_weights->dptr<PredType>(), outside_weights->dptr<PredType>(),
      beta, scale, pred_diff_blob->mut_dptr<PredType>());
}

template<typename PredType, typename LabelType>
struct SmoothL1LossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t instance_num, const int64_t instance_dim,
                      const PredType* prediction, const LabelType* label,
                      const PredType* inside_weights, const PredType* outside_weights,
                      const float beta, const float scale, PredType* loss) {
    int64_t elem_cnt = instance_num * instance_dim;
    for (int i = 0; i < elem_cnt; i++) {
      PredType x = inside_weights[i] * (prediction[i] - label[i]);
      PredType abs_x = abs(x);
      if (abs_x < beta) {
        loss[i] = 0.5 * x * x / beta;
      } else {
        loss[i] = abs_x - 0.5 * beta;
      }
      loss[i] *= scale * outside_weights[i];
    }
  }
  static void Backward(DeviceCtx* ctx, const int64_t instance_num, const int64_t instance_dim,
                       const PredType* prediction, const LabelType* label,
                       const PredType* inside_weights, const PredType* outside_weights,
                       const float beta, const float scale, PredType* in_diff) {
    int64_t elem_cnt = instance_num * instance_dim;
    for (int i = 0; i < elem_cnt; i++) {
      PredType x = inside_weights[i] * (prediction[i] - label[i]);
      PredType abs_x = abs(x);
      if (abs_x < beta) {
        in_diff[i] = x / beta;
      } else {
        in_diff[i] = x > 0 ? 1 : -1;
      }
      in_diff[i] *= scale;
    }
  }
};

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf& SmoothL1LossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.smooth_l1_loss_conf().loss_conf();
}

namespace {

Kernel* CreateSmoothL1LossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SMOOTH_L1_LOSS_KERNEL_ENTRY(device_type, pred_type_pair, label_type_pair)                  \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new SmoothL1LossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair),                  \
                                   OF_PP_PAIR_FIRST(label_type_pair)>();                           \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SMOOTH_L1_LOSS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                                kernel_conf.smooth_l1_loss_conf().loss_conf().prediction_type(),
                                kernel_conf.smooth_l1_loss_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kSmoothL1LossConf, CreateSmoothL1LossKernel);

}  // namespace oneflow
