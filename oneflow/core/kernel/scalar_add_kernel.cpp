#include "oneflow/core/kernel/scalar_add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ScalarAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  T scalar_operand = 0;
  const auto& conf = this->op_conf().scalar_add_conf();
  if (IsIntegral<T>::value) {
    scalar_operand = static_cast<T>(conf.int_operand());
  } else if (IsFloating<T>::value) {
    scalar_operand = static_cast<T>(conf.float_operand());
  } else {
    UNIMPLEMENTED();
  }
  KernelUtil<device_type, T>::AddByScalar(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                          in_blob->dptr<T>(), scalar_operand,
                                          out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ScalarAddKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  Memcpy<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), out_diff_blob->dptr<T>(),
                      out_diff_blob->ByteSizeOfDataContentField());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kScalarAddConf, ScalarAddKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
