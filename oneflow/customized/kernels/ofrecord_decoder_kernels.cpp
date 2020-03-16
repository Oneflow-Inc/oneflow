#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

class OFRecordImageDecoderRandomCropKernel final : public user_op::OpKernel {
 public:
  OFRecordImageDecoderRandomCropKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {
    int64_t seed = ctx->GetAttr<int64_t>("seed");
    if (seed == -1) { seed = NewRandomSeed(); }
    CHECK(seed >= 0);
    random_generator_.reset(new RandomGenerator<DeviceType::kCPU>(seed, ctx->device_ctx()));
  }
  OFRecordImageDecoderRandomCropKernel() = default;
  ~OFRecordImageDecoderRandomCropKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    random_generator_->Uniform<float>(out_blob->shape().elem_cnt(), 0.0, 1.0,
                                      out_blob->mut_dptr<float>());
  }

  std::unique_ptr<RandomGenerator<DeviceType::kCPU>> random_generator_;
};

REGISTER_USER_KERNEL("OFRecordImageDecoderRandomCrop")
    .SetCreateFn([](user_op::KernelInitContext* ctx) {
      return new OFRecordImageDecoderRandomCropKernel(ctx);
    })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && out_tensor->data_type() == DataType::kFloat) {
        return true;
      }
      return false;
    });

}  // namespace oneflow
