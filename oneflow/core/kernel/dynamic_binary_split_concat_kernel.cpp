#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class DynamicBinarySplitKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicBinarySplitKernel);
  DynamicBinarySplitKernel() = default;
  ~DynamicBinarySplitKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx &,
                          std::function<Blob *(const std::string &)>) const override;
  void ForwardShape(const KernelCtx &ctx,
                    std::function<Blob *(const std::string &)> BnInOp2Blob) const override;
};

void DynamicBinarySplitKernel::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  const Blob *in_blob = BnInOp2Blob("in");
  int64_t remain_size = in_blob->AlignedTotalByteSize();
  const char *header_ptr = in_blob->header_ptr();
  int64_t offset = 0;
  CHECK_EQ(header_ptr + in_blob->blob_desc().ByteSizeOfBlobHeader(), in_blob->dptr<char>());
  for (const auto &output_bn : this->op_attribute().output_bns()) {
    Blob *out_blob = BnInOp2Blob(output_bn);
    int64_t out_size = out_blob->shape_view().elem_cnt();
    if (out_size > 0) { memcpy(out_blob->mut_dptr(), header_ptr + offset, out_size); }
    remain_size -= out_size;
    offset += out_size;
  }
  CHECK_EQ(remain_size, 0);
  CHECK_EQ(offset, in_blob->AlignedTotalByteSize());
}

void DynamicBinarySplitKernel::ForwardShape(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {
  const Blob *in_blob = BnInOp2Blob("in");
  int64_t remain_size = in_blob->AlignedTotalByteSize();
  for (const auto &output_bn : this->op_attribute().output_bns()) {
    Blob *out_blob = BnInOp2Blob(output_bn);
    int64_t out_size = out_blob->ByteSizeOfBlobBody();
    if (remain_size >= out_size) {
      remain_size -= out_size;
      out_blob->mut_shape_view()->set_shape(Shape({out_size}));
    } else {
      out_blob->mut_shape_view()->set_shape(Shape({0}));
    }
  }
  if (remain_size > 0) {
    int32_t out_num = this->op_attribute().output_bns().size();
    const std::string &last_obn = this->op_attribute().output_bns(out_num - 1);
    Blob *last_out_blob = BnInOp2Blob(last_obn);
    CHECK(last_out_blob->IsBodyEmpty());
    last_out_blob->mut_shape_view()->set_shape(Shape({remain_size}));
  }
}

// REGISTER_KERNEL(OperatorConf::kDynamicBinarySplitConf, DynamicBinarySplitKernel);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kDynamicBinarySplitConf, DeviceType::kCPU, char,
                                      DynamicBinarySplitKernel);

}  // namespace oneflow
