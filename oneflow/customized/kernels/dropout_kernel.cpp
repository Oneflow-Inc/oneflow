#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"
#include "oneflow/customized/kernels/random_mask_generator.h"

namespace oneflow {

namespace {

template<typename T>
void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x, const int8_t* mask,
                  T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = x[i] * static_cast<T>(mask[i]) * scale; }
}

template<typename T>
class DropoutKernelCPU final : public user_op::OpKernel {
 public:
  DropoutKernelCPU() = default;
  ~DropoutKernelCPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const float scale = ctx->Attr<float>("scale");
    MaskAndScale<T>(ctx->device_ctx(), in->shape().elem_cnt(), scale, in->dptr<T>(),
                    mask->dptr<int8_t>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL_CPU(dtype)                                                      \
  REGISTER_USER_KERNEL("dropout")                                                               \
      .SetCreateFn<DropoutKernelCPU<dtype>>()                                                   \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU                             \
                       & user_op::HobDataType("out", 0) == GetDataType<dtype>::value)           \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_KERNEL_CPU(float)
REGISTER_DROPOUT_KERNEL_CPU(double)

template<typename T>
class DropoutGradKernelCPU final : public user_op::OpKernel {
 public:
  DropoutGradKernelCPU() = default;
  ~DropoutGradKernelCPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float scale = ctx->Attr<float>("scale");
    MaskAndScale<T>(ctx->device_ctx(), dy->shape().elem_cnt(), scale, dy->dptr<T>(),
                    mask->dptr<int8_t>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_GRAD_KERNEL_CPU(dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                          \
      .SetCreateFn<DropoutGradKernelCPU<dtype>>()                                               \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU                             \
                       & user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)            \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_GRAD_KERNEL_CPU(float)
REGISTER_DROPOUT_GRAD_KERNEL_CPU(double)

template<DeviceType device_type>
class RandomMaskLikeKernel final : public user_op::OpKernel {
 public:
  RandomMaskLikeKernel() = default;
  ~RandomMaskLikeKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    int64_t seed = ctx->Attr<int64_t>("seed");
    return std::make_shared<OpKernelStateWrapper<RandomMaskGenerator<device_type>>>(seed);
  }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* like = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t elem_cnt = like->shape().elem_cnt();
    int8_t* mask = out->mut_dptr<int8_t>();
    auto* random_mask_generator =
        dynamic_cast<OpKernelStateWrapper<RandomMaskGenerator<device_type>>*>(state);
    CHECK_NOTNULL(random_mask_generator);
    random_mask_generator->Mutable()->Generate(ctx->device_ctx(), elem_cnt,
                                               ctx->Attr<float>("rate"), mask);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RANDOM_MASK_LIKE_KERNEL(device)   \
  REGISTER_USER_KERNEL("random_mask_like")         \
      .SetCreateFn<RandomMaskLikeKernel<device>>() \
      .SetIsMatchedHob(user_op::HobDeviceType() == device);

REGISTER_RANDOM_MASK_LIKE_KERNEL(DeviceType::kCPU)
REGISTER_RANDOM_MASK_LIKE_KERNEL(DeviceType::kGPU)

}  // namespace
}  // namespace oneflow
