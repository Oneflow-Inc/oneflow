#include "oneflow/core/kernel/const_scalar_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConstScalarKernel<device_type, T>::VirtualKernelInit(const ParallelContext*) {
  output_inited_ = false;
  if (IsIntegralDataType(this->kernel_conf().data_type())) {
    CHECK(this->op_conf().const_scalar_conf().has_int_val());
    const_val_ = static_cast<T>(this->op_conf().const_scalar_conf().int_val());
  } else if (IsFloatingDataType(this->kernel_conf().data_type())) {
    CHECK(this->op_conf().const_scalar_conf().has_float_val());
    const_val_ = static_cast<T>(this->op_conf().const_scalar_conf().float_val());
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void ConstScalarKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (output_inited_) { return; }
  KernelUtil<device_type, T>::Set(ctx.device_ctx, const_val_, BnInOp2Blob("out")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConstScalarConf, ConstScalarKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
