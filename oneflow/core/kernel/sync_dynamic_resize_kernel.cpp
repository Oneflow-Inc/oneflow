#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

class SyncDynamicResizeKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SyncDynamicResizeKernel);
  SyncDynamicResizeKernel() = default;
  ~SyncDynamicResizeKernel() override = default;

 private:
  bool IsKernelLaunchSynchronized() const override { return false; }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const SyncDynamicResizeOpConf& conf = this->op_conf().sync_dynamic_resize_conf();
    CHECK_EQ(conf.axis(), 0);
    std::shared_ptr<int32_t> size_on_cpu(new int32_t);
    const Blob* in = BnInOp2Blob("in");
    const Blob* size = BnInOp2Blob("size");
    Blob* out = BnInOp2Blob("out");
    AutoMemcpy(ctx.device_ctx, out->mut_dptr(), in->dptr(), in->ByteSizeOfBlobBody(),
               out->mem_case(), in->mem_case());
    AutoMemcpy(ctx.device_ctx, size_on_cpu.get(), size->dptr(), sizeof(int32_t), MakeHostMemCase(),
               size->mem_case());
    ctx.device_ctx->AddCallBack(
        [out, size_on_cpu, conf]() { out->dense_shape_mut_view()->Set(conf.axis(), *size_on_cpu); },
        this->kernel_conf().op_attribute().op_conf().name());
  }
};

REGISTER_KERNEL_WITH_NOTHING(OperatorConf::kSyncDynamicResizeConf, SyncDynamicResizeKernel);

}  // namespace oneflow
