#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/data/coco_data_reader.h"

namespace oneflow {

namespace {

class COCOReaderWrapper final : public user_op::OpKernelState {
 public:
  explicit COCOReaderWrapper(user_op::KernelInitContext* ctx) : reader_(ctx) {}
  ~COCOReaderWrapper() = default;

  void Read(user_op::KernelComputeContext* ctx) { reader_.Read(ctx); }

 private:
  data::COCODataReader reader_;
};

class COCOReaderKernel final : public user_op::OpKernel {
 public:
  COCOReaderKernel() = default;
  ~COCOReaderKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    std::shared_ptr<user_op::OpKernelState> reader(new COCOReaderWrapper(ctx));
    return reader;
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* reader = dynamic_cast<COCOReaderWrapper*>(state);
    reader->Read(ctx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

REGISTER_USER_KERNEL("COCOReader")
    .SetCreateFn<COCOReaderKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU
                     & user_op::HobDataType("image", 0) == DataType::kTensorBuffer
                     & user_op::HobDataType("image_id", 0) == DataType::kInt64
                     & user_op::HobDataType("image_size", 0) == DataType::kInt32
                     & user_op::HobDataType("gt_bbox", 0) == DataType::kTensorBuffer
                     & user_op::HobDataType("gt_label", 0) == DataType::kTensorBuffer
                     & user_op::HobDataType("gt_segm", 0) == DataType::kTensorBuffer
                     & user_op::HobDataType("gt_segm_index", 0) == DataType::kTensorBuffer);

}  // namespace oneflow
