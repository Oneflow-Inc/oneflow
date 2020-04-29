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
  COCODataReader reader_;
};

}  // namespace

class COCOReaderKernel final : public user_op::OpKernel {
 public:
  COCOReaderKernel() = default;
  ~COCOReaderKernel() override = default;

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
};

REGISTER_USER_KERNEL("COCOReader")
    .SetCreateFn<COCOReaderKernel>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* image_desc = ctx.TensorDesc4ArgNameAndIndex("image", 0);
      const user_op::TensorDesc* image_id_desc = ctx.TensorDesc4ArgNameAndIndex("image_id", 0);
      const user_op::TensorDesc* image_size_desc = ctx.TensorDesc4ArgNameAndIndex("image_size", 0);
      const user_op::TensorDesc* bbox_desc = ctx.TensorDesc4ArgNameAndIndex("gt_bbox", 0);
      const user_op::TensorDesc* label_desc = ctx.TensorDesc4ArgNameAndIndex("gt_label", 0);
      const user_op::TensorDesc* segm_desc = ctx.TensorDesc4ArgNameAndIndex("gt_segm", 0);
      const user_op::TensorDesc* segm_offset_desc =
          ctx.TensorDesc4ArgNameAndIndex("gt_segm_offset", 0);
      return ctx.device_type() == DeviceType::kCPU
             && image_desc->data_type() == DataType::kTensorBuffer
             && image_id_desc->data_type() == DataType::kInt64
             && image_size_desc->data_type() == DataType::kInt32
             && bbox_desc->data_type() == DataType::kTensorBuffer
             && label_desc->data_type() == DataType::kTensorBuffer
             && segm_desc->data_type() == DataType::kTensorBuffer
             && segm_offset_desc->data_type() == DataType::kTensorBuffer;
    });

}  // namespace oneflow
