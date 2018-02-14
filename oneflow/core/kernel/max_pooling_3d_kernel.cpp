#include "oneflow/core/kernel/max_pooling_3d_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MaxPooling3DKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Pooling3DKernelUtil<device_type, T>::Forward(ctx, in_blob, out_blob,
                                               this->pooling_3d_ctx());
}

template<DeviceType device_type, typename T>
void MaxPooling3DKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  Pooling3DKernelUtil<device_type, T>::Backward(ctx, out_diff_blob, out_blob,
                                                in_blob, in_diff_blob,
                                                this->pooling_3d_ctx());
}

template<DeviceType device_type, typename T>
const Pooling3DKernelConf&
MaxPooling3DKernel<device_type, T>::GetPooling3DKernelConf() const {
  return this->kernel_conf().max_pooling_3d_conf().pooling_3d_conf();
}

namespace {

namespace MaxPooling1D {
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling1DConf, MaxPooling3DKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}

namespace MaxPooling2D {
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling2DConf, MaxPooling3DKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}

namespace MaxPooling3D {
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling3DConf, MaxPooling3DKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}

}  // namespace

}  // namespace oneflow
