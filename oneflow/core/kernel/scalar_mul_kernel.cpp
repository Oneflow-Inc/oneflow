#include "oneflow/core/kernel/scalar_mul_kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ScalarMulKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  T scalar_operand = static_cast<T>(0);
  const auto& conf = this->op_conf().scalar_mul_conf();
  if (conf.has_int_operand()) {
    scalar_operand = static_cast<T>(conf.int_operand());
  } else if (conf.has_float_operand()) {
    scalar_operand = static_cast<T>(conf.float_operand());
  } else {
    UNIMPLEMENTED();
  }
  NewKernelUtil<device_type>::MulByScalar(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                          in_blob->dptr<T>(), scalar_operand,
                                          out_blob->mut_dptr<T>());
}

#define REGISTER_SCALAR_MUL_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kScalarMulConf, DeviceType::kCPU, dtype, \
                                        ScalarMulKernel<DeviceType::kCPU, dtype>);             \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kScalarMulConf, DeviceType::kGPU, dtype, \
                                        ScalarMulKernel<DeviceType::kGPU, dtype>);

REGISTER_SCALAR_MUL_KERNEL(float);
REGISTER_SCALAR_MUL_KERNEL(double);
REGISTER_SCALAR_MUL_KERNEL(int32_t);

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kScalarMulConf, DeviceType::kGPU, float16,
                                      ScalarMulKernel<DeviceType::kGPU, float16>);

#undef REGISTER_SCALAR_MUL_KERNEL

}  // namespace oneflow
