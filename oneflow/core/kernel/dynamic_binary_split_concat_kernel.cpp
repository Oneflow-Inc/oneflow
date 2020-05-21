#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace {

int64_t GetBlobDynamicTotalSize(const Blob* blob) {
  // blob dynamic total byte size = STATIC header byte size + DYNAMIC body byte size
  return blob->blob_desc().ByteSizeOfBlobHeader() + blob->ByteSizeOfBlobBody();
}

int64_t GetBlobDynamicBodySize(const Blob* blob) {
  // Attention! dynamic body size MUST get from Blob::ByteSizeOfBlobBody()
  return blob->ByteSizeOfBlobBody();
}

int64_t GetBlobStaticBodySize(const Blob* blob) {
  // Attention! static body size MUST get from RtBlobDesc::ByteSizeOfBlobBody()
  return blob->blob_desc().ByteSizeOfBlobBody();
}

}  // namespace

class DynamicBinarySplitKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicBinarySplitKernel);
  DynamicBinarySplitKernel() = default;
  ~DynamicBinarySplitKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardShape(const KernelCtx& ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

void DynamicBinarySplitKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  CHECK(in_blob->blob_desc().is_dynamic());
  int64_t in_blob_total_dynamic_size = GetBlobDynamicTotalSize(in_blob);
  int64_t remain_size = in_blob_total_dynamic_size;
  const char* header_ptr = in_blob->header_ptr();
  int64_t offset = 0;
  CHECK_EQ(header_ptr + in_blob->blob_desc().ByteSizeOfBlobHeader(), in_blob->dptr<char>());
  for (const auto& output_bn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(output_bn);
    int64_t dynamic_out_size = GetBlobDynamicBodySize(out_blob);
    if (dynamic_out_size > 0) {
      memcpy(out_blob->mut_dptr(), header_ptr + offset, dynamic_out_size);
    }
    remain_size -= dynamic_out_size;
    offset += dynamic_out_size;
  }
  CHECK_EQ(remain_size, 0);
  CHECK_EQ(offset, in_blob_total_dynamic_size);
}

void DynamicBinarySplitKernel::ForwardShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  int64_t remain_size = GetBlobDynamicTotalSize(in_blob);
  for (const auto& output_bn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(output_bn);
    int64_t static_out_size = GetBlobStaticBodySize(out_blob);
    if (remain_size >= static_out_size) {
      remain_size -= static_out_size;
      out_blob->mut_shape_view()->set_shape(Shape({static_out_size}));
    } else {
      out_blob->mut_shape_view()->set_shape(Shape({0}));
    }
  }
  if (remain_size > 0) {
    int32_t out_num = this->op_attribute().output_bns().size();
    const std::string& last_obn = this->op_attribute().output_bns(out_num - 1);
    Blob* last_out_blob = BnInOp2Blob(last_obn);
    CHECK(last_out_blob->IsBodyEmpty());
    last_out_blob->mut_shape_view()->set_shape(Shape({remain_size}));
  }
}

class DynamicBinaryConcatKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicBinaryConcatKernel);
  DynamicBinaryConcatKernel() = default;
  ~DynamicBinaryConcatKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardShape(const KernelCtx& ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

void DynamicBinaryConcatKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  int64_t offset = 0;
  for (const auto& ibn : this->op_attribute().input_bns()) {
    Blob* in_blob = BnInOp2Blob(ibn);
    int64_t dynamic_in_size = GetBlobDynamicBodySize(in_blob);
    if (dynamic_in_size > 0) {
      memcpy(out_blob->mut_contiguous_header_ptr() + offset, in_blob->dptr(), dynamic_in_size);
    }
    offset += dynamic_in_size;
  }
  CHECK_GT(offset, 0);
  CHECK_EQ(offset, GetBlobDynamicTotalSize(out_blob));
}

void DynamicBinaryConcatKernel::ForwardShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // DO NOTHING
}

// REGISTER_KERNEL(OperatorConf::kDynamicBinarySplitConf, DynamicBinarySplitKernel);
// REGISTER_KERNEL(OperatorConf::kDynamicBinaryConcatConf, DynamicBinaryConcatKernel);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kDynamicBinarySplitConf, DeviceType::kCPU,
                            DynamicBinarySplitKernel);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kDynamicBinaryConcatConf, DeviceType::kCPU,
                            DynamicBinaryConcatKernel);

}  // namespace oneflow
