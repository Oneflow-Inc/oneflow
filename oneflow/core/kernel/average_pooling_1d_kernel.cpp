#include "oneflow/core/kernel/average_pooling_1d_kernel.h"
#include "oneflow/core/kernel/average_pooling_3d_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void AveragePooling1DKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  AveragePooling3DKernelUtil<device_type, T>::Forward(ctx, in_blob, out_blob,
                                                      this->pooling_3d_ctx());
}

template<DeviceType device_type, typename T>
void AveragePooling1DKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  AveragePooling3DKernelUtil<device_type, T>::Backward(
      ctx, out_diff_blob, in_diff_blob, this->pooling_3d_ctx());
}

template<DeviceType device_type, typename T>
const Pooling1DKernelConf&
AveragePooling1DKernel<device_type, T>::GetPooling1DKernelConf() const {
  return this->kernel_conf().average_pooling_1d_conf().pooling_1d_conf();
}

template<DeviceType device_type, typename T>
const PbMessage& AveragePooling1DKernel<device_type, T>::GetPooling1DOpConf()
    const {
  return this->op_conf().average_pooling_1d_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAveragePooling1DConf,
                           AveragePooling1DKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
