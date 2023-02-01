/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> NewBroadcastElementwiseBinaryPrimitive(
    Context* ctx, ep::primitive::BinaryOp op) {
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  const int64_t ndims = y->shape().NumAxes();
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), op, x->data_type(), y->data_type(), ndims);
}

template<ep::primitive::BinaryOp op>
auto BroadcastElementwiseBinaryPrimitiveExists() {
  return hob::make_custom("BroadcastElementwiseBinaryPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewBroadcastElementwiseBinaryPrimitive(&ctx, op).operator bool();
                          });
}

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseBinary>
NewBroadcastElementwiseAttrBinaryPrimitive(Context* ctx, ep::primitive::BinaryOp op) {
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
  const int64_t ndims = dy->shape().NumAxes();
  Scalar value;
  if (ctx->template Attr<bool>("has_int_operand")) {
    value = Scalar(ctx->template Attr<int64_t>("int_operand"));
  } else if (ctx->template Attr<bool>("has_float_operand")) {
    value = Scalar(ctx->template Attr<double>("float_operand"));
  } else {
    UNIMPLEMENTED();
  }
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), op, x->data_type(), dy->data_type(), ndims, value);
}

template<ep::primitive::BinaryOp op>
auto BroadcastElementwiseAttrBinaryPrimitiveExists() {
  return hob::make_custom(
      "BroadcastElementwiseBinaryAttrPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
        return NewBroadcastElementwiseAttrBinaryPrimitive(&ctx, op).operator bool();
      });
}

}  // namespace

template<ep::primitive::BinaryOp op>
class ScalarMathKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ScalarMathKernel() = default;
  ~ScalarMathKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Scalar value;
    if (ctx->Attr<bool>("has_int_operand")) {
      value = Scalar(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      value = Scalar(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      const bool is_add_sub_0 =
          (op == ep::primitive::BinaryOp::kAdd || op == ep::primitive::BinaryOp::kSub)
          && value.Value<double>() == 0.0;
      const bool is_mul_div_1 =
          (op == ep::primitive::BinaryOp::kMul || op == ep::primitive::BinaryOp::kDiv)
          && value.Value<double>() == 1.0;
      if ((is_add_sub_0 || is_mul_div_1) && in->dptr() == out->dptr()) { return; }
      std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> primitive =
          NewBroadcastElementwiseBinaryPrimitive(ctx, op);
      CHECK(primitive);
      primitive->Launch(ctx->stream(), in->shape_view().NumAxes(), in->shape_view().ptr(),
                        in->dptr(), value, out->mut_dptr());
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<ep::primitive::BinaryOp op>
class ScalarReverseMathKernel final : public user_op::OpKernel {
 public:
  ScalarReverseMathKernel() = default;
  ~ScalarReverseMathKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Scalar value;
    if (ctx->Attr<bool>("has_int_operand")) {
      value = Scalar(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      value = Scalar(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> primitive =
          NewBroadcastElementwiseBinaryPrimitive(ctx, op);
      CHECK(primitive);
      primitive->Launch(ctx->stream(), value, in->shape_view().NumAxes(), in->shape_view().ptr(),
                        in->dptr(), out->mut_dptr());
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define SCALAR_MATH_SEQ                                                       \
  OF_PP_MAKE_TUPLE_SEQ("scalar_add", ep::primitive::BinaryOp::kAdd)           \
  OF_PP_MAKE_TUPLE_SEQ("scalar_mul", ep::primitive::BinaryOp::kMul)           \
  OF_PP_MAKE_TUPLE_SEQ("scalar_div", ep::primitive::BinaryOp::kDiv)           \
  OF_PP_MAKE_TUPLE_SEQ("scalar_floordiv", ep::primitive::BinaryOp::kFloorDiv) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_truncdiv", ep::primitive::BinaryOp::kTruncDiv) \
  OF_PP_MAKE_TUPLE_SEQ("scalar_fmod", ep::primitive::BinaryOp::kFmod)         \
  OF_PP_MAKE_TUPLE_SEQ("scalar_pow", ep::primitive::BinaryOp::kPow)

#define REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(op_name, binary_op)          \
  REGISTER_USER_KERNEL(op_name)                                                      \
      .SetCreateFn<ScalarMathKernel<binary_op>>()                                    \
      .SetIsMatchedHob((BroadcastElementwiseBinaryPrimitiveExists<binary_op>()))     \
      .SetInplaceProposalFn(                                                         \
          [](const user_op::InferContext& ctx,                                       \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> { \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));        \
            return Maybe<void>::Ok();                                                \
          });

OF_PP_FOR_EACH_TUPLE(REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL, SCALAR_MATH_SEQ)

#define REGISTER_UNARY_MATH_SCALAR_REVERSE_ELEMWISE_USER_KERNEL(op_name, binary_op)                \
  REGISTER_USER_KERNEL(op_name).SetCreateFn<ScalarReverseMathKernel<binary_op>>().SetIsMatchedHob( \
      (BroadcastElementwiseBinaryPrimitiveExists<binary_op>()));

REGISTER_UNARY_MATH_SCALAR_REVERSE_ELEMWISE_USER_KERNEL("scalar_reverse_pow",
                                                        ep::primitive::BinaryOp::kPow)

template<ep::primitive::BinaryOp op>
class ScalarPowGradKernel final : public user_op::OpKernel {
 public:
  ScalarPowGradKernel() = default;
  ~ScalarPowGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t elem_cnt = dx_tensor->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> primitive =
          NewBroadcastElementwiseAttrBinaryPrimitive(ctx, op);
      CHECK(primitive);
      primitive->Launch(ctx->stream(), x_tensor->shape_view().NumAxes(),
                        x_tensor->shape_view().ptr(), x_tensor->dptr(),
                        dy_tensor->shape_view().NumAxes(), dy_tensor->shape_view().ptr(),
                        dy_tensor->dptr(), dx_tensor->mut_dptr());
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BINARY_MATH_WITH_ATTR_ELEMWISE_USER_KERNEL(op_name, binary_op)                \
  REGISTER_USER_KERNEL(op_name).SetCreateFn<ScalarPowGradKernel<binary_op>>().SetIsMatchedHob( \
      (BroadcastElementwiseAttrBinaryPrimitiveExists<binary_op>()));

REGISTER_BINARY_MATH_WITH_ATTR_ELEMWISE_USER_KERNEL("scalar_pow_grad",
                                                    ep::primitive::BinaryOp::kScalarBasePowerGrad);
REGISTER_BINARY_MATH_WITH_ATTR_ELEMWISE_USER_KERNEL("scalar_reverse_pow_grad",
                                                    ep::primitive::BinaryOp::kScalarExpPowerGrad);

}  // namespace oneflow
