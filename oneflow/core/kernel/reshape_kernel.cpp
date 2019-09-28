#include "oneflow/core/kernel/reshape_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReshapeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReshapeConf, ReshapeKernel);

template<DeviceType device_type>
class ExpandDimsKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExpandDimsKernel);
  ExpandDimsKernel() = default;
  ~ExpandDimsKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
  }
};

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kExpandDimsConf, DeviceType::kGPU,
                            ExpandDimsKernel<DeviceType::kGPU>);

template<DeviceType device_type>
class SqueezeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SqueezeKernel);
  SqueezeKernel() = default;
  ~SqueezeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
  }
};

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kSqueezeConf, DeviceType::kGPU,
                            SqueezeKernel<DeviceType::kGPU>);

}  // namespace oneflow
