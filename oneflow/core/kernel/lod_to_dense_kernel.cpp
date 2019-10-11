#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace {

void CheckSizeAndCopyBlob(DeviceCtx *ctx, Blob *dst, const Blob *src) {
  CHECK_EQ(src->ByteSizeOfBlobBody(), dst->ByteSizeOfBlobBody());
  dst->CopyDataContentFrom(ctx, src);
}

}  // namespace

template<DeviceType device_type>
class LoDToDenseKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LoDToDenseKernel);
  LoDToDenseKernel() = default;
  ~LoDToDenseKernel() = default;

 private:
  void ForwardLoD(const KernelCtx &, std::function<Blob *(const std::string &)>) const override {}
  void ForwardDataContent(const KernelCtx &,
                          std::function<Blob *(const std::string &)>) const override;
};

template<DeviceType device_type>
void LoDToDenseKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  CheckSizeAndCopyBlob(ctx.device_ctx, BnInOp2Blob("out"), BnInOp2Blob("in"));
  std::this_thread::sleep_for(std::chrono::seconds(this->op_conf().sleep_conf().seconds()));
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kLodToDenseConf, DeviceType::kCPU,
                            LoDToDenseKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kLodToDenseConf, DeviceType::kGPU,
                            LoDToDenseKernel<DeviceType::kGPU>);

}  // namespace oneflow
