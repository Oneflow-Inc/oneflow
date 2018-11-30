#include "oneflow/core/kernel/loss_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename PredType>
void LossKernel<device_type, PredType>::SetLossInstanceNum(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  CHECK_GE(this->op_attribute().input_bns().size(), 2);
  // already did CheckSameDim0ValidNum in Kernel::Forward
  const int64_t dim0_valid_num_sum =
      BnInOp2Blob(this->op_attribute().input_bns(0))->CalcDim0ValidNumSum();
  KernelUtil<device_type, PredType>::Set(ctx.device_ctx, static_cast<PredType>(dim0_valid_num_sum),
                                         BnInOp2Blob("loss_instance_num")->mut_dptr<PredType>());
  CHECK(BnInOp2Blob(GenDiffBn("prediction"))->has_loss_instance_num_field());
  BnInOp2Blob(GenDiffBn("prediction"))->set_loss_instance_num(dim0_valid_num_sum);
}

template<DeviceType device_type, typename PredType>
void LossKernel<device_type, PredType>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  VirtualLossForwardDataContent(ctx, BnInOp2Blob);
  SetLossInstanceNum(ctx, BnInOp2Blob);

  const LossKernelConf& conf = GetLossKernelConf(this->kernel_conf());
  int64_t n = BnInOp2Blob("prediction")->shape().At(0);
  Blob* weight_blob = BnInOp2Blob("weight");
  // backward
  // predict_diff *= weight
  Blob* prediction_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  if (prediction_diff_blob != nullptr) {
    PredType* prediction_diff = prediction_diff_blob->mut_dptr<PredType>();
    if (weight_blob != nullptr) {
      PredType* weight = weight_blob->mut_dptr<PredType>();
      if (weight_blob->shape().elem_cnt() == n) {
        const int64_t m = prediction_diff_blob->shape().Count(1);
        NdarrayUtil<device_type, PredType>::template BroadcastApply<BinaryFuncMul>(
            ctx.device_ctx, XpuVarNdarray<PredType>({n, m}, prediction_diff),
            XpuVarNdarray<const PredType>({n, 1}, weight),
            XpuVarNdarray<const PredType>({n, m}, prediction_diff));
      } else if (weight_blob->shape().elem_cnt() == 1) {
        KernelUtil<device_type, PredType>::Scal(ctx.device_ctx, n, weight, prediction_diff, 1);
      } else {
        UNIMPLEMENTED();
      }
    } else if (conf.weight_scalar() > 1.0 || conf.weight_scalar() < 1.0) {
      KernelUtil<device_type, PredType>::Scal(
          ctx.device_ctx, n, static_cast<PredType>(conf.weight_scalar()), prediction_diff, 1);
    }
  }

  // compute reduction_coefficient
  Blob* reduction_blob = BnInOp2Blob("reduction_coefficient");
  if (reduction_blob != nullptr) {
    CHECK(weight_blob != nullptr);
    LossKernelUtil<device_type, PredType>::ComputeReductionCoefficient(
        ctx.device_ctx, n, weight_blob->shape().elem_cnt(), weight_blob->dptr<PredType>(),
        reduction_blob->mut_dptr<PredType>(), conf.reduction());
  }
}

template<DeviceType device_type, typename PredType>
void LossKernel<device_type, PredType>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("loss")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("prediction"));
}

template<DeviceType device_type, typename PredType>
void LossKernel<device_type, PredType>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob(GenDiffBn("prediction"))->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("prediction"));
}

template<DeviceType device_type, typename PredType>
void LossKernel<device_type, PredType>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob(GenDiffBn("prediction"))
      ->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("prediction"));
  BnInOp2Blob("loss")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("prediction"));
}

template<DeviceType device_type, typename PredType>
void LossKernel<device_type, PredType>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
struct LossKernelUtil<DeviceType::kCPU, T> {
  static void ComputeReductionCoefficient(DeviceCtx* ctx, int64_t data_num, int64_t weight_length,
                                          const T* weight, T* reduction, LossReductionType type) {
    switch (type) {
      case kSumOverOne: *reduction = 1.0; break;
      case kSumOverWeight: {
        if (weight_length == data_num) {
          *reduction = 0;
          for (size_t i = 0; i < weight_length; ++i) { (*reduction) += weight[i]; }
        } else if (weight_length == 1) {
          *reduction = (*weight) * data_num;
        } else {
          UNIMPLEMENTED();
        }
        break;
      }
      case kSumOverN: *reduction = 1.0 * data_num; break;
      case kSumOverNonZeroWeight: {
        if (weight_length == data_num) {
          *reduction = 0;
          for (size_t i = 0; i < weight_length; ++i) {
            if (weight[i] > 0.0) { (*reduction) += 1.0; }
          }
        } else if (weight_length == 1) {
          *reduction = 1.0 * data_num;
        } else {
          UNIMPLEMENTED();
        }
        break;
      }
      default: UNIMPLEMENTED();
    }
  }
};

#define MAKE_LOSS_KERNEL_UTIL_ENTRY(type_cpp, type_proto) \
  template struct LossKernelUtil<DeviceType::kCPU, type_cpp>;

OF_PP_FOR_EACH_TUPLE(MAKE_LOSS_KERNEL_UTIL_ENTRY, FLOATING_DATA_TYPE_SEQ)

#define MAKE_LOSS_ENTRY(device_type, data_type_pair) \
  template class LossKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_LOSS_ENTRY, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
