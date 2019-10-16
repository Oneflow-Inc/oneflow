#include <chrono>
#include <thread>
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
class SleepKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SleepKernel);
  SleepKernel() = default;
  ~SleepKernel() = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx &,
                          std::function<Blob *(const std::string &)>) const override;
  void ForwardLoD(const KernelCtx &ctx,
                  std::function<Blob *(const std::string &)> BnInOp2Blob) const override;
};

template<DeviceType device_type>
void SleepKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  CheckSizeAndCopyBlob(ctx.device_ctx, BnInOp2Blob("out"), BnInOp2Blob("in"));
  LOG(INFO) << this->op_conf().name() << " starts sleeping for "
            << this->op_conf().sleep_conf().seconds() << " seconds";
  std::this_thread::sleep_for(std::chrono::seconds(this->op_conf().sleep_conf().seconds()));
}

template<DeviceType device_type>
void SleepKernel<device_type>::ForwardLoD(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  const Blob *in_blob = BnInOp2Blob("in");
  Blob *out_blob = BnInOp2Blob("out");
  out_blob->tree_lod_mut_view().UpdateLoD(in_blob->tree_lod_view().lod_tree());
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kSleepConf, SleepKernel);

}  // namespace oneflow
