#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

namespace {

template<typename T>
void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x, const int8_t* mask,
                  T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = x[i] * static_cast<T>(mask[i]) * scale; }
}

void GenMask(DeviceCtx* ctx, const int64_t n, float threshold, const float* random_tmp,
             int8_t* mask) {
  for (int64_t i = 0; i < n; ++i) { mask[i] = random_tmp[i] > threshold; }
}

}  // namespace

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
    const float scale = ctx->GetAttr<float>("scale");
    MaskAndScale<T>(ctx->device_ctx(), in->shape().elem_cnt(), scale, in->dptr<T>(),
                    mask->dptr<int8_t>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL_CPU(dtype)                                                      \
  REGISTER_USER_KERNEL("dropout")                                                               \
      .SetCreateFn<DropoutKernelCPU<dtype>>()                                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);           \
        return ctx.device_type() == DeviceType::kCPU                                            \
               && y_desc->data_type() == GetDataType<dtype>::value;                             \
      })                                                                                        \
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
    const float scale = ctx->GetAttr<float>("scale");
    MaskAndScale<T>(ctx->device_ctx(), dy->shape().elem_cnt(), scale, dy->dptr<T>(),
                    mask->dptr<int8_t>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_GRAD_KERNEL_CPU(dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                          \
      .SetCreateFn<DropoutGradKernelCPU<dtype>>()                                               \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);           \
        return ctx.device_type() == DeviceType::kCPU                                            \
               && dx_desc->data_type() == GetDataType<dtype>::value;                            \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_GRAD_KERNEL_CPU(float)
REGISTER_DROPOUT_GRAD_KERNEL_CPU(double)

class RandomMaskLikeKernelCPU final : public user_op::OpKernel {
 public:
  RandomMaskLikeKernelCPU() = default;
  ~RandomMaskLikeKernelCPU() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    int64_t seed = ctx->GetAttr<int64_t>("seed");
    return std::make_shared<OpKernelStateWrapper<RandomGenerator<DeviceType::kCPU>>>(
        seed, ctx->device_ctx());
  }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* like = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    int64_t elem_cnt = like->shape().elem_cnt();
    int8_t* mask = out->mut_dptr<int8_t>();
    float* random_tmp = tmp_buffer->mut_dptr<float>();

    auto* random_generator =
        dynamic_cast<OpKernelStateWrapper<RandomGenerator<DeviceType::kCPU>>*>(state);
    random_generator->Mutable()->Uniform(elem_cnt, random_tmp);

    GenMask(ctx->device_ctx(), elem_cnt, ctx->GetAttr<float>("rate"), random_tmp, mask);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("random_mask_like")
    .SetCreateFn<RandomMaskLikeKernelCPU>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      return ctx.device_type() == DeviceType::kCPU;
    })
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      const Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
      const size_t tmp_buffer_bytes = GetCudaAlignedSize(like_shape->elem_cnt() * sizeof(float));
      return tmp_buffer_bytes;
    });

}  // namespace oneflow
