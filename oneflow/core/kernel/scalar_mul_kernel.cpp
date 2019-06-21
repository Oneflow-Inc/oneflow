#include "oneflow/core/kernel/scalar_mul_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ScalarMulKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  T scalar_operand = 0;
  const auto& conf = this->op_conf().scalar_mul_conf();
  if (conf.has_int_operand()) {
    scalar_operand = static_cast<T>(conf.int_operand());
  } else if (conf.has_float_operand()) {
    scalar_operand = static_cast<T>(conf.float_operand());
  } else {
    UNIMPLEMENTED();
  }
  KernelUtil<device_type, T>::MulByScalarPara(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                              in_blob->dptr<T>(), scalar_operand,
                                              out_blob->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kScalarMulConf, ScalarMulKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
