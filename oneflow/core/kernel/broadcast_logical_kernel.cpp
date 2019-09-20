#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastEqualKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastEqualKernel);
  BroadcastEqualKernel() = default;
  ~BroadcastEqualKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& kernel_ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* a_blob = BnInOp2Blob("a");
    const Blob* b_blob = BnInOp2Blob("b");
    Blob* out_blob = BnInOp2Blob("out");
    size_t num_axes = out_blob->shape().NumAxes();
    NdarrayUtil<device_type, T>::BroadcastEQ(
        kernel_ctx.device_ctx, XpuVarNdarray<int8_t>(out_blob, num_axes),
        XpuVarNdarray<const T>(a_blob, num_axes), XpuVarNdarray<const T>(b_blob, num_axes));
  }
};

#define REGISTER_BROADCAST_COMPARISION_KERNEL(type, dev, dtype)                  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, dev, dtype, \
                                        Broadcast##type##Kernel<dev, dtype>)

REGISTER_BROADCAST_COMPARISION_KERNEL(Equal, DeviceType::kGPU, float);
REGISTER_BROADCAST_COMPARISION_KERNEL(Equal, DeviceType::kGPU, double);
REGISTER_BROADCAST_COMPARISION_KERNEL(Equal, DeviceType::kCPU, float);
REGISTER_BROADCAST_COMPARISION_KERNEL(Equal, DeviceType::kCPU, double);
}  // namespace oneflow
