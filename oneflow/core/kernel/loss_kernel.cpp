#include "oneflow/core/kernel/loss_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename PredType>
void LossKernel<device_type, PredType>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  VirtualLossForwardDataContent(ctx, BnInOp2Blob);

  const LossKernelConf& conf = GetLossKernelConf(this->kernel_conf());
  int64_t n = BnInOp2Blob("prediction")->shape().At(0);
  Blob* weight_blob = BnInOp2Blob("weight");

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
void LossKernel<device_type, PredType>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
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
                                          const T* weight, T* reduction, ScalarReductionType type) {
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
