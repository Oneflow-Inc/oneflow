#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void LossKernel<device_type, PredType, LabelType>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  VirtualLossForwardDataContent(ctx, BnInOp2Blob);
  LossKernelConf& conf = GetLossKernelConf(kernel_conf());
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  const int64_t n = prediction_blob->shape().At(0);
  // backward
  // predict_diff *= weight
  Blob* prediction_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  if (prediction_diff_blob != nullptr) {
      PredType* prediction_diff = prediction_diff_blob->mut_dptr<PredType>();
    if(conf.need_weight_blob()) {
      Blob* weight_blob = BnInOp2Blob("weight");
      PredType* weight = weight_blob->mut_dptr<PredType>();
      if (weight_blob->shape().elem_cnt() == n) {
        KernelUtil<devict_type, PredType>::Mul(ctx.device_ctx, n, weight, prediction_diff, prediction_diff);
      } else if (weight_blob->shape().elem_cnt() == 1) {
     KernelUtil<device_type, PredType>::Scal(ctx.device_ctx, n, weight, prediction_diff, 1);
      } else {
        UNIMPLEMENTED(); 
      }
    } else if (conf.weight_scalar() != 1.0) {
     PredType weight_scalar = static_cast<PredType>(conf.weight_scalar());
     KernelUtil<device_type, PredType>::Scal(ctx.device_ctx, n, &weight_scalar, prediction_diff, 1);
    }
  }

  // compute reduction_coefficient
  PredType* reduction_coefficient = BnInOp2Blob("reduction_coefficient")->mut_dptr<PredType>();
  if (conf.need_weight_blob()){
  
  }
}

template<DeviceType device_type, typename PredType, typename LabelType>
void LossKernel<device_type, PredType, LabelType>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("loss")->CopyDataIdFrom(ctx.device_ctx,
                                      BnInOp2Blob("prediction"));
}

template<DeviceType device_type, typename PredType, typename LabelType>
void LossKernel<device_type, PredType, LabelType>::ForwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob(GenDiffBn("prediction"))
      ->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("prediction"));
}

#define MAKE_LOSS_ENTRY(device_type, data_type_pair, label_type_pair)      \
  template class LossKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair), \
                            OF_PP_PAIR_FIRST(label_type_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_LOSS_ENTRY, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
