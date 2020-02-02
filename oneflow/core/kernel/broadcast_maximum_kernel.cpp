#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastMaximumKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMaximumKernel);
  BroadcastMaximumKernel() = default;
  ~BroadcastMaximumKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void BroadcastMaximumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a = BnInOp2Blob("a");
  const Blob* b = BnInOp2Blob("b");
  Blob* out = BnInOp2Blob("out");
  size_t num_axes = out->shape().NumAxes();
  NdarrayUtil<device_type, T>::BroadcastMax(ctx.device_ctx, XpuVarNdarray<T>(out, num_axes),
                                            XpuVarNdarray<const T>(a, num_axes),
                                            XpuVarNdarray<const T>(b, num_axes));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastMaximumConf, BroadcastMaximumKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
