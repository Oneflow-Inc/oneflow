#include "oneflow/core/kernel/one_hot_kernel.h"

namespace oneflow {

template <DeviceType device_type, typename T>
const PbMessage& OneHotKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().one_hot_conf();
}

template <DeviceType device_type, typename T>
void OneHotKernel<device_type, T>::ForwardDataContent(const KernelCtx& ctx,
                        std::function<Blob*(const std::string&)> BnInOp2Blob) const {


}

template <DeviceType device_type, typename T>
void OneHotKernel<device_type, T>::BackwardDataContent(const KernelCtx& ctx,
                                                       std::function<Blob*(const std::string&)> BnInOp2Blob) const {

}

}
