#include "oneflow/core/kernel/smooth_l1_loss_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_loss_kernel.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SmoothL1LossKernel<device_type, PredType, LabelType>::
    VirtualLossForwardDataContent(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf&
SmoothL1LossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.smooth_l1_loss_conf().loss_conf();
}

namespace {

Kernel* CreateSmoothL1LossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SMOOTH_L1_LOSS_KERNEL_ENTRY(device_type, pred_type_pair,                \
                                                       label_type_pair)                            \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new SmoothL1LossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair), \
                                                    OF_PP_PAIR_FIRST(label_type_pair)>();          \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SMOOTH_L1_LOSS_KERNEL_ENTRY,
                                       DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                 kernel_conf.smooth_l1_loss_conf().loss_conf().prediction_type(),
                 kernel_conf.smooth_l1_loss_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kSmoothL1LossConf, CreateSmoothL1LossKernel);

}  // namespace oneflow
