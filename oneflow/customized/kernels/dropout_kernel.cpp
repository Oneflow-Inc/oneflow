#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/dropout_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class DropoutKernel final : public user_op::OpKernel {
 public:
  DropoutKernel() = default;
  ~DropoutKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const float scale = ctx->GetAttr<float>("scale");
    DropoutKernelUtil<device_type, T>::MaskAndScale(
        ctx->device_ctx(), in->shape().elem_cnt(), scale, in->dptr<T>(), mask->dptr<int8_t>(), out->mut_dptr<T>());
  };
};

#define REGISTER_DROPOUT_KERNEL(device, dtype)                                                     \
  REGISTER_USER_KERNEL("dropout")                                                                  \
      .SetCreateFn<DropoutKernel<device, dtype>>()                                                 \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);           \
        return ctx.device_type() == device && y_desc->data_type() == GetDataType<dtype>::value; \
      });

REGISTER_DROPOUT_KERNEL(DeviceType::kCPU, float)
REGISTER_DROPOUT_KERNEL(DeviceType::kCPU, double)
REGISTER_DROPOUT_KERNEL(DeviceType::kGPU, float)
REGISTER_DROPOUT_KERNEL(DeviceType::kGPU, double)
REGISTER_DROPOUT_KERNEL(DeviceType::kGPU, float16)

template<DeviceType device_type, typename T>
class DropoutGradKernel final : public user_op::OpKernel {
 public:
  DropoutGradKernel() = default;
  ~DropoutGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float scale = ctx->GetAttr<float>("scale");
    DropoutKernelUtil<device_type, T>::MaskAndScale(
        ctx->device_ctx(), dy->shape().elem_cnt(), scale, dy->dptr<T>(), mask->dptr<int8_t>(), dx->mut_dptr<T>());
  };
};

#define REGISTER_DROPOUT_GRAD_KERNEL(device, dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                              \
      .SetCreateFn<DropoutGradKernel<device, dtype>>()                                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                               \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);            \
        return ctx.device_type() == device && dx_desc->data_type() == GetDataType<dtype>::value; \
      });

REGISTER_DROPOUT_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_DROPOUT_GRAD_KERNEL(DeviceType::kCPU, double)
REGISTER_DROPOUT_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_DROPOUT_GRAD_KERNEL(DeviceType::kGPU, double)
REGISTER_DROPOUT_GRAD_KERNEL(DeviceType::kGPU, float16)

template<DeviceType device_type>
class RandomMaskLikeKernel final : public user_op::OpKernel {
 public:
  RandomMaskLikeKernel() = default;
  ~RandomMaskLikeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* like = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
        
    int64_t elem_cnt = out->shape().elem_cnt();
    float* random_tmp = temp_buffer->mut_dptr<float>();
    int8_t* mask = out->mut_dptr<int8_t>();

    random_generator_->Uniform(elem_cnt, random_tmp);
    RandomMaskLikeKernelUtil<device_type>::GenMask(ctx.device_ctx, elem_cnt,
                                                   this->op_conf().random_mask_like_conf().rate(),
                                                   random_tmp, mask);
  };
};

#define REGISTER_RANDOM_MASK_LIKE_KERNEL(device)                                                     \
  REGISTER_USER_KERNEL("random_mask_like")                                                                  \
      .SetCreateFn<RandomMaskLikeKernel<device, dtype>>()                                                 \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        return ctx.device_type() == device; \
      });

REGISTER_RANDOM_MASK_LIKE_KERNEL(DeviceType::kCPU)
REGISTER_RANDOM_MASK_LIKE_KERNEL(DeviceType::kGPU)

}  // namespace

}  // namespace oneflow
