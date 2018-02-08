#include "oneflow/core/kernel/average_pooling_3d_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void AveragePooling3DKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  AveragePooling3DKernelUtil<device_type, T>::Forward(ctx, in_blob, out_blob,
                                                      this->pooling_3d_ctx());
}

template<DeviceType device_type, typename T>
void AveragePooling3DKernel<device_type, T>::BackwardDataContent(
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
const Pooling3DKernelConf&
AveragePooling3DKernel<device_type, T>::GetPooling3DKernelConf() const {
  return this->kernel_conf().average_pooling_3d_conf().pooling_3d_conf();
}

template<typename T>
class AveragePooling3DKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling3DKernelUtil);
  AveragePooling3DKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const Blob* in_blob, Blob* out_blob,
                      const Pooling3DCtx& pooling_ctx) {
    // eigen
    TODO();
  }

  static void Backward(const KernelCtx& ctx, const Blob* out_diff_blob,
                       Blob* in_diff_blob, const Pooling3DCtx& pooling_ctx) {
    TODO();
  }
};

namespace {

namespace AveragePooling1D {
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAveragePooling1DConf,
                           AveragePooling3DKernel, ARITHMETIC_DATA_TYPE_SEQ);
}

namespace AveragePooling2D {
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAveragePooling2DConf,
                           AveragePooling3DKernel, ARITHMETIC_DATA_TYPE_SEQ);
}

namespace AveragePooling3D {
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAveragePooling3DConf,
                           AveragePooling3DKernel, ARITHMETIC_DATA_TYPE_SEQ);
}

}  // namespace

}  // namespace oneflow
