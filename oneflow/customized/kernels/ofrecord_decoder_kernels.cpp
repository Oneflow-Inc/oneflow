#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/customized/image/random_crop_generator.h"

namespace oneflow {

class OFRecordImageDecoderRandomCropKernel final : public user_op::OpKernel {
 public:
  OFRecordImageDecoderRandomCropKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {
    int32_t num_attempts = ctx->GetAttr<int32_t>("num_attempts");
    CHECK(num_attempts >= 1);
    std::vector<float> random_aspect_ratio =
        ctx->GetAttr<std::vector<float>>("random_aspect_ratio");
    CHECK(random_aspect_ratio.size() == 2 && 0 < random_aspect_ratio.at(0)
          && random_aspect_ratio.at(0) <= random_aspect_ratio.at(1));
    std::vector<float> random_area = ctx->GetAttr<std::vector<float>>("random_area");
    CHECK(random_area.size() == 2 && 0 < random_area.at(0)
          && random_area.at(0) <= random_area.at(1));
    const user_op::TensorDesc* out_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
    CHECK(out_tensor_desc->shape().NumAxes() == 1);
    int64_t batch_size = out_tensor_desc->shape().At(0);
    CHECK(batch_size > 0);
    int64_t seed = ctx->GetAttr<int64_t>("seed");
    if (seed == -1) { seed = NewRandomSeed(); }
    CHECK(seed >= 0);
    std::seed_seq seq{seed};
    std::vector<int64_t> seeds(batch_size);
    seq.generate(seeds.begin(), seeds.end());

    crop_window_generators_.resize(batch_size);
    for (int32_t i = 0; i < batch_size; ++i) {
      std::shared_ptr<RandomCropGenerator> random_crop_generator(new RandomCropGenerator(
          {random_aspect_ratio.at(0), random_aspect_ratio.at(1)},
          {random_area.at(0), random_area.at(1)}, seeds.at(i), num_attempts));
      crop_window_generators_.at(i) = std::bind(&RandomCropGenerator::GenerateCropWindow,
                                                random_crop_generator, std::placeholders::_1);
    }
  }
  OFRecordImageDecoderRandomCropKernel() = default;
  ~OFRecordImageDecoderRandomCropKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    // random_generator_->Uniform<float>(out_blob->shape().elem_cnt(), 0.0, 1.0,
    //                                  out_blob->mut_dptr<float>());
  }

  std::vector<CropWindowGenerator> crop_window_generators_;
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
