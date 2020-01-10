#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/nvtx3/nvToolsExt.h"

namespace oneflow {

namespace {

void CheckSizeAndCopyBlob(DeviceCtx *ctx, Blob *dst, const Blob *src) {
  CHECK_EQ(src->ByteSizeOfBlobBody(), dst->ByteSizeOfBlobBody());
  dst->CopyDataContentFrom(ctx, src);
}

}  // namespace

template<DeviceType device_type>
class NvtxRangePushKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NvtxRangePushKernel);
  NvtxRangePushKernel() = default;
  ~NvtxRangePushKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx &ctx,
                          std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    CheckSizeAndCopyBlob(ctx.device_ctx, BnInOp2Blob("out"), BnInOp2Blob("in"));
    nvtxRangePush(this->op_conf().nvtx_range_push_conf().msg().c_str());
  }
  void ForwardLoD(const KernelCtx &ctx,
                  std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    const Blob *in_blob = BnInOp2Blob("in");
    Blob *out_blob = BnInOp2Blob("out");
    out_blob->tree_lod_mut_view().UpdateLoD(in_blob->tree_lod_view().lod_tree());
  }
};

template<DeviceType device_type>
class NvtxRangePopKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NvtxRangePopKernel);
  NvtxRangePopKernel() = default;
  ~NvtxRangePopKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx &ctx,
                          std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    CheckSizeAndCopyBlob(ctx.device_ctx, BnInOp2Blob("out"), BnInOp2Blob("in"));
    nvtxRangePop();
  }
  void ForwardLoD(const KernelCtx &ctx,
                  std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    const Blob *in_blob = BnInOp2Blob("in");
    Blob *out_blob = BnInOp2Blob("out");
    out_blob->tree_lod_mut_view().UpdateLoD(in_blob->tree_lod_view().lod_tree());
  }
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kNvtxRangePushConf, NvtxRangePushKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kNvtxRangePopConf, NvtxRangePopKernel);

}  // namespace oneflow
