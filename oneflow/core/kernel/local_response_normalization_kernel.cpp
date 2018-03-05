#include "oneflow/core/kernel/local_response_normalization_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LocalResponseNormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type, typename T>
void LocalResponseNormalizationKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

#define INSTANTIATE_LOCAL_RESPONSE_NORMALIZATION_KERNEL(type_cpp, type_proto) \
  template class LocalResponseNormalizationKernel<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_LOCAL_RESPONSE_NORMALIZATION_KERNEL,
                     ARITHMETIC_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalResponseNormalizationConf,
                           LocalResponseNormalizationKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
