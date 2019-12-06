#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace {

void CheckSizeAndCopyBlob(DeviceCtx *ctx, Blob *dst, const Blob *src) {
  dst->CopyDataContentFrom(ctx, src);
}

}  // namespace

template<DeviceType device_type>
class DistributeSplitKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeSplitKernel);
  DistributeSplitKernel() = default;
  ~DistributeSplitKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx &,
                          std::function<Blob *(const std::string &)>) const override;
  Blob *GetOutBlob(std::function<Blob *(const std::string &)> BnInOp2Blob) const;
};

template<DeviceType device_type>
void DistributeSplitKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  CheckSizeAndCopyBlob(ctx.device_ctx, GetOutBlob(BnInOp2Blob), BnInOp2Blob("in"));
}

template<DeviceType device_type>
Blob *DistributeSplitKernel<device_type>::GetOutBlob(
    std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  Blob *out_blob = nullptr;
  FOR_RANGE(int, i, 0, this->op_attribute().output_bns().size()) {
    Blob *cur_blob = BnInOp2Blob(this->op_attribute().output_bns().Get(i));
    if (cur_blob != nullptr && cur_blob != out_blob) {
      CHECK_ISNULL(out_blob);
      out_blob = cur_blob;
    }
  }
  return out_blob;
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDistributeSplitConf, DistributeSplitKernel);

}  // namespace oneflow
