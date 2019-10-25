#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastAddKernel);
  BroadcastAddKernel() = default;
  ~BroadcastAddKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* a_blob = BnInOp2Blob("a");
    const Blob* b_blob = BnInOp2Blob("b");
    Blob* out_blob = BnInOp2Blob("out");
    size_t num_axes = out_blob->shape().NumAxes();
    NdarrayUtil<device_type, T>::BroadcastAdd(ctx.device_ctx, XpuVarNdarray<T>(out_blob, num_axes),
                                              XpuVarNdarray<const T>(a_blob, num_axes),
                                              XpuVarNdarray<const T>(b_blob, num_axes));
  }
};

#define REGISTER_BROADCAST_ADD_KERNEL_ENTRY(dev, dtype)                              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcastAddConf, dev, dtype, \
                                        BroadcastAddKernel<dev, dtype>)

#define REGISTER_BROADCAST_ADD_KERNEL(dtype)                    \
  REGISTER_BROADCAST_ADD_KERNEL_ENTRY(DeviceType::kCPU, dtype); \
  REGISTER_BROADCAST_ADD_KERNEL_ENTRY(DeviceType::kGPU, dtype);

REGISTER_BROADCAST_ADD_KERNEL(float);
REGISTER_BROADCAST_ADD_KERNEL(double);
REGISTER_BROADCAST_ADD_KERNEL(int32_t);
REGISTER_BROADCAST_ADD_KERNEL(int64_t);

REGISTER_BROADCAST_ADD_KERNEL_ENTRY(DeviceType::kGPU, float16);

#undef REGISTER_BROADCAST_ADD_KERNEL
#undef REGISTER_BROADCAST_ADD_KERNEL_ENTRY

}  // namespace oneflow
