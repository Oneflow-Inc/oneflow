#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/dropout_kernel.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"
#include "oneflow/core/kernel/random_generator.h"

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
    DropoutKernelUtil<device_type, T>::MaskAndScale(ctx->device_ctx(), in->shape().elem_cnt(),
                                                    scale, in->dptr<T>(), mask->dptr<int8_t>(),
                                                    out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL(device, dtype)                                                  \
  REGISTER_USER_KERNEL("dropout").SetCreateFn<DropoutKernel<device, dtype>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                                \
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
    DropoutKernelUtil<device_type, T>::MaskAndScale(ctx->device_ctx(), dy->shape().elem_cnt(),
                                                    scale, dy->dptr<T>(), mask->dptr<int8_t>(),
                                                    dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_GRAD_KERNEL(device, dtype)                                              \
  REGISTER_USER_KERNEL("dropout_grad")                                                           \
      .SetCreateFn<DropoutGradKernel<device, dtype>>()                                           \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                               \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);            \
        return ctx.device_type() == device && dx_desc->data_type() == GetDataType<dtype>::value; \
      });

// OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DROPOUT_KERNEL, DEVICE_TYPE_SEQ,
//                                  ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

// OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DROPOUT_GRAD_KERNEL, DEVICE_TYPE_SEQ,
//                                  ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)
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
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    int64_t seed = ctx->GetAttr<int64_t>("seed");
    return std::make_shared<OpKernelStateWrapper<RandomGenerator<device_type>>>(seed,
                                                                                ctx->device_ctx());
  }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* like = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    int64_t elem_cnt = like->shape().elem_cnt();
    int8_t* mask = out->mut_dptr<int8_t>();
    float* random_tmp = tmp_buffer->mut_dptr<float>();

    auto* random_generator =
        dynamic_cast<OpKernelStateWrapper<RandomGenerator<device_type>>*>(state);
    random_generator->Mutable()->Uniform(elem_cnt, random_tmp);

    RandomMaskLikeKernelUtil<device_type>::GenMask(ctx->device_ctx(), elem_cnt,
                                                   ctx->GetAttr<float>("rate"), random_tmp, mask);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RANDOM_MASK_LIKE_KERNEL(device)                                            \
  REGISTER_USER_KERNEL("random_mask_like")                                                  \
      .SetCreateFn<RandomMaskLikeKernel<device>>()                                          \
      .SetIsMatchedPred(                                                                    \
          [](const user_op::KernelRegContext& ctx) { return ctx.device_type() == device; }) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);                    \
        const size_t tmp_buffer_bytes =                                                     \
            GetCudaAlignedSize(like_shape->elem_cnt() * sizeof(float));                     \
        return tmp_buffer_bytes;                                                            \
      });

REGISTER_RANDOM_MASK_LIKE_KERNEL(DeviceType::kCPU)
REGISTER_RANDOM_MASK_LIKE_KERNEL(DeviceType::kGPU)

}  // namespace

}  // namespace oneflow
