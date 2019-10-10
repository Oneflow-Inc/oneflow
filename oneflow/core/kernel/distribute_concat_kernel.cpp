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
class DistributeConcatKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeConcatKernel);
  DistributeConcatKernel() = default;
  ~DistributeConcatKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx &,
                          std::function<Blob *(const std::string &)>) const override;
  void ForwardLoD(const KernelCtx &ctx,
                  std::function<Blob *(const std::string &)> BnInOp2Blob) const override;

  const Blob *GetInBlob(std::function<Blob *(const std::string &)> BnInOp2Blob) const;
};

template<DeviceType device_type>
void DistributeConcatKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  CheckSizeAndCopyBlob(ctx.device_ctx, BnInOp2Blob("out"), GetInBlob(BnInOp2Blob));
}

template<DeviceType device_type>
void DistributeConcatKernel<device_type>::ForwardLoD(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  const Blob *in_blob = GetInBlob(BnInOp2Blob);
  Blob *out_blob = BnInOp2Blob("out");
  out_blob->tree_lod_mut_view().UpdateLoD(in_blob->tree_lod_view().lod_tree());
}

template<DeviceType device_type>
const Blob *DistributeConcatKernel<device_type>::GetInBlob(
    std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  const Blob *in_blob = nullptr;
  FOR_RANGE(int, i, 0, this->op_attribute().input_bns().size()) {
    const Blob *cur_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(i));
    if (cur_blob != nullptr && cur_blob != in_blob) {
      CHECK_ISNULL(in_blob);
      in_blob = cur_blob;
    }
  }
  return in_blob;
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDistributeConcatConf, DistributeConcatKernel);

}  // namespace oneflow
