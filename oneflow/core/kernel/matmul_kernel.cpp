#include "oneflow/core/kernel/matmul_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type, typename T>
const PbMessage& MatmulKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().matmul_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMatmulConf, MatmulKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
