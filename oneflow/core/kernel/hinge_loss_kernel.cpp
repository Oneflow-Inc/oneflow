#include "oneflow/core/kernel/hinge_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void HingeLossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  const Blob* label_blob = BnInOp2Blob("label");
  Blob* loss_blob = BnInOp2Blob("loss");
  Blob* tmp_diff_blob = BnInOp2Blob("tmp_diff");
  const int64_t data_num = prediction_blob->shape().At(0);
  const int64_t pre_dim = prediction_blob->shape().Count(1);
  const LabelType* label = label_blob->dptr<LabelType>();
  const PredType* pred = prediction_blob->dptr<PredType>();
  PredType* loss = loss_blob->mut_dptr<PredType>();
  const OperatorConf& op_conf = this->op_conf();
  tmp_diff_blob->CopyDataContentFrom(ctx.device_ctx, prediction_blob);
  PredType* tmp_diff = tmp_diff_blob->mut_dptr<PredType>();
  // forward
  HingeLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, data_num, pre_dim, pred, label, op_conf, tmp_diff, loss);
  // if predict_diff_blob is not null, then do backward
  Blob* prediction_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  if (prediction_diff_blob != nullptr) {
    PredType* pred_diff = prediction_diff_blob->mut_dptr<PredType>();
    HingeLossKernelUtil<device_type, PredType, LabelType>::Backward(
        ctx.device_ctx, data_num, pre_dim, tmp_diff, label, op_conf, pred_diff);
  }
}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf& HingeLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.hinge_loss_conf().loss_conf();
}

template<typename PredType, typename LabelType>
struct HingeLossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t data_num, const int64_t pre_dim,
                      const PredType* pred, const LabelType* label, const OperatorConf& op_conf,
                      PredType* tmp_diff, PredType* loss) {
    // transfor sign of each pred according to label
    for (int64_t i = 0; i < data_num; ++i) {
      tmp_diff[i * pre_dim + static_cast<int64_t>(label[i])] *= -1;
    }
    // compute diff of each dim
    for (int64_t i = 0; i < data_num * pre_dim; ++i) {
      tmp_diff[i] = (1 + tmp_diff[i]) > 0 ? (1 + tmp_diff[i]) : 0;
    }
    switch (op_conf.hinge_loss_conf().norm()) {
      case L1:
        KernelUtil<DeviceType::kCPU, PredType>::RowSum(ctx, data_num, pre_dim, tmp_diff, loss);
        /*for (int64_t i = 0; i < data_num; ++i) {
          KernelUtil<DeviceType::kCPU, PredType>::Sum(ctx, pre_dim, tmp_diff + i * pre_dim,
                                                      loss + i);
        }*/
        break;
      case L2:
        for (int64_t i = 0; i < data_num; ++i) {
          KernelUtil<DeviceType::kCPU, PredType>::Dot(ctx, pre_dim, tmp_diff + i * pre_dim, 1,
                                                      tmp_diff + i * pre_dim, 1, loss + i);
        }
        break;
      default: LOG(FATAL) << "Invalid norm method in " << op_conf.name();
    }
  }

  static void Backward(DeviceCtx* ctx, const int64_t data_num, const int64_t pre_dim,
                       const PredType* tmp_diff, const LabelType* label, const OperatorConf op_conf,
                       PredType* pred_diff) {
    for (int64_t i = 0; i < data_num * pre_dim; ++i) { pred_diff[i] = (tmp_diff[i] > 0); }
    for (int64_t i = 0; i < data_num; ++i) {
      pred_diff[i * pre_dim + static_cast<int64_t>(label[i])] *= -1;
    }
    switch (op_conf.hinge_loss_conf().norm()) {
      case L1: break;
      case L2:
        for (int64_t i = 0; i < data_num * pre_dim; ++i) {
          pred_diff[i] = 2 * tmp_diff[i] * pred_diff[i];
        }
        break;
      default: LOG(FATAL) << "Invalid norm method in " << op_conf.name();
    }
  }
};

namespace {

Kernel* CreateHingeLossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define HINGE_LOSS_KERNEL_ENTRY(device_type, pred_type_pair, label_type_pair)                      \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new HingeLossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair),                     \
                                OF_PP_PAIR_FIRST(label_type_pair)>();                              \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(HINGE_LOSS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                                kernel_conf.hinge_loss_conf().loss_conf().prediction_type(),
                                kernel_conf.hinge_loss_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kHingeLossConf, CreateHingeLossKernel);

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                       \
  template struct HingeLossKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                      OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
