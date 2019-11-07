#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class DynamicReshapeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicReshapeKernel);
  DynamicReshapeKernel() = default;
  ~DynamicReshapeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
  }
};

#define REGISTER_DYNAMIC_KERNELS(name, dev)                                           \
  REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kDynamic##name##Conf, DeviceType::k##dev, \
                              Dynamic##name##Kernel<DeviceType::k##dev>);

REGISTER_DYNAMIC_KERNELS(Reshape, CPU);
REGISTER_DYNAMIC_KERNELS(Reshape, GPU);

template<DeviceType device_type>
class DynamicReshapeLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicReshapeLikeKernel);
  DynamicReshapeLikeKernel() = default;
  ~DynamicReshapeLikeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("x");
    Blob* out_blob = BnInOp2Blob("y");
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
  }
};

REGISTER_DYNAMIC_KERNELS(ReshapeLike, CPU);
REGISTER_DYNAMIC_KERNELS(ReshapeLike, GPU);

}  // namespace oneflow
