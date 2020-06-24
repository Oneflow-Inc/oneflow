#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/data/ofrecord_data_reader.h"

namespace oneflow {

namespace {

class OFRecordReaderWrapper final : public user_op::OpKernelState {
 public:
  explicit OFRecordReaderWrapper(user_op::KernelInitContext* ctx) : reader_(ctx) {}
  ~OFRecordReaderWrapper() = default;

  void Read(user_op::KernelComputeContext* ctx) { reader_.Read(ctx); }

 private:
  data::OFRecordDataReader reader_;
};

}  // namespace

class OFRecordReaderKernel final : public user_op::OpKernel {
 public:
  OFRecordReaderKernel() = default;
  ~OFRecordReaderKernel() override = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    std::shared_ptr<OFRecordReaderWrapper> reader(new OFRecordReaderWrapper(ctx));
    return reader;
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* reader = dynamic_cast<OFRecordReaderWrapper*>(state);
    reader->Read(ctx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("OFRecordReader")
    .SetCreateFn<OFRecordReaderKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU
                     & user_op::HobDataType("out", 0) == DataType::kOFRecord);

}  // namespace oneflow
