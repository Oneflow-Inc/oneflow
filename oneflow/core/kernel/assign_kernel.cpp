#include "oneflow/core/kernel/assign_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void AssignKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x_blob = BnInOp2Blob("x");
  Blob* y_blob = BnInOp2Blob("y");
  // Skip memory copy if inplace
  if (y_blob->dptr<T>() != x_blob->dptr<T>()) {
    Memcpy<device_type>(ctx.device_ctx, y_blob->mut_dptr<T>(), x_blob->dptr<T>(),
                        y_blob->shape().elem_cnt());
  }
}

template<DeviceType device_type, typename T>
const PbMessage& AssignKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().assign_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAssignConf, AssignKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
