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
class DistributeCloneKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeCloneKernel);
  DistributeCloneKernel() = default;
  ~DistributeCloneKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx &,
                          std::function<Blob *(const std::string &)>) const override;
  void ForwardDenseShape(const KernelCtx &ctx,
                         std::function<Blob *(const std::string &)> BnInOp2Blob) const override;
  void ForwardLoD(const KernelCtx &ctx,
                  std::function<Blob *(const std::string &)> BnInOp2Blob) const override;

  Blob *GetOutBlob(std::function<Blob *(const std::string &)> BnInOp2Blob) const;
};

template<DeviceType device_type>
void DistributeCloneKernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  CheckSizeAndCopyBlob(ctx.device_ctx, GetOutBlob(BnInOp2Blob), BnInOp2Blob("in"));
}

template<DeviceType device_type>
void DistributeCloneKernel<device_type>::ForwardLoD(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  const Blob *in_blob = BnInOp2Blob("in");
  Blob *out_blob = GetOutBlob(BnInOp2Blob);
  out_blob->tree_lod_mut_view().UpdateLoD(in_blob->tree_lod_view().lod_tree());
}

template<DeviceType device_type>
void DistributeCloneKernel<device_type>::ForwardDenseShape(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  Blob *out_blob = GetOutBlob(BnInOp2Blob);
  out_blob->dense_shape_mut_view()->set_shape(BnInOp2Blob("in")->shape());
}

template<DeviceType device_type>
Blob *DistributeCloneKernel<device_type>::GetOutBlob(
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

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDistributeCloneConf, DistributeCloneKernel);

}  // namespace oneflow
