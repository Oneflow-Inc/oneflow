#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device, typename T>
class ScalarAddByTensorKernel final : public user_op::OpKernel {
 public:
  ScalarAddByTensorKernel() = default;
  ~ScalarAddByTensorKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    NewKernelUtil<device>::AddByScalarPtr(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                          scalar->dptr<T>(), y->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCALAR_ADD_BY_TENSOR_KERNEL(device, dtype_pair)                                \
  REGISTER_USER_KERNEL("scalar_add_by_tensor")                                                  \
      .SetCreateFn<ScalarAddByTensorKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()             \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);             \
        return ctx.device_type() == device                                                      \
               && x_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);                         \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_ADD_BY_TENSOR_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

template<DeviceType device, typename T>
class ScalarSubByTensorKernel final : public user_op::OpKernel {
 public:
  ScalarSubByTensorKernel() = default;
  ~ScalarSubByTensorKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    NewKernelUtil<device>::SubByScalarPtr(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                          scalar->dptr<T>(), y->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCALAR_SUB_BY_TENSOR_KERNEL(device, dtype_pair)                                \
  REGISTER_USER_KERNEL("scalar_sub_by_tensor")                                                  \
      .SetCreateFn<ScalarSubByTensorKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()             \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);             \
        return ctx.device_type() == device                                                      \
               && x_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);                         \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_SUB_BY_TENSOR_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

template<DeviceType device, typename T>
class ScalarMulByTensorKernel final : public user_op::OpKernel {
 public:
  ScalarMulByTensorKernel() = default;
  ~ScalarMulByTensorKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    NewKernelUtil<device>::MulByScalarPtr(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                          scalar->dptr<T>(), y->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCALAR_MUL_BY_TENSOR_KERNEL(device, dtype_pair)                                \
  REGISTER_USER_KERNEL("scalar_mul_by_tensor")                                                  \
      .SetCreateFn<ScalarMulByTensorKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()             \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);             \
        return ctx.device_type() == device                                                      \
               && x_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);                         \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_MUL_BY_TENSOR_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

template<DeviceType device, typename T>
class ScalarDivByTensorKernel final : public user_op::OpKernel {
 public:
  ScalarDivByTensorKernel() = default;
  ~ScalarDivByTensorKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    NewKernelUtil<device>::DivByScalarPtr(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                          scalar->dptr<T>(), y->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCALAR_DIV_BY_TENSOR_KERNEL(device, dtype_pair)                                \
  REGISTER_USER_KERNEL("scalar_div_by_tensor")                                                  \
      .SetCreateFn<ScalarDivByTensorKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()             \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);             \
        return ctx.device_type() == device                                                      \
               && x_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);                         \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_DIV_BY_TENSOR_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
