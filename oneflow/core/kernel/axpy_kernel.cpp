#include "oneflow/core/kernel/axpy_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void AxpyKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x_blob = BnInOp2Blob("x");
  Blob* y_blob = BnInOp2Blob("y");
  T alpha = this->op_conf().axpy_conf().alpha();
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, y_blob->shape().elem_cnt(), alpha,
                                   x_blob->dptr<T>(), 1, y_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
const PbMessage& AxpyKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().axpy_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAxpyConf, AxpyKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
