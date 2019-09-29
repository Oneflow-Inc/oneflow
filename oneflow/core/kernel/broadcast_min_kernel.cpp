#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastMinKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMinKernel);
  BroadcastMinKernel() = default;
  ~BroadcastMinKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* a_blob = BnInOp2Blob("a");
    const Blob* b_blob = BnInOp2Blob("b");
    Blob* out_blob = BnInOp2Blob("out");
    size_t num_axes = out_blob->shape().NumAxes();
    NdarrayUtil<device_type, T>::BroadcastMin(ctx.device_ctx, XpuVarNdarray<T>(out_blob, num_axes),
                                              XpuVarNdarray<const T>(a_blob, num_axes),
                                              XpuVarNdarray<const T>(b_blob, num_axes));
  }
};

#define REGISTER_BROADCAST_MIN_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcastMinConf, DeviceType::kCPU, dtype, \
                                        BroadcastMinKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcastMinConf, DeviceType::kGPU, dtype, \
                                        BroadcastMinKernel<DeviceType::kGPU, dtype>)

REGISTER_BROADCAST_MIN_KERNEL(float);
REGISTER_BROADCAST_MIN_KERNEL(double);
REGISTER_BROADCAST_MIN_KERNEL(int8_t);
REGISTER_BROADCAST_MIN_KERNEL(int32_t);
REGISTER_BROADCAST_MIN_KERNEL(int64_t);

}  // namespace oneflow
