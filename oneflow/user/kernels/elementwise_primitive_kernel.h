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
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/ep/include/primitive/elementwise_unary.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/ep/include/primitive/binary_op.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

class UnaryPrimitiveKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnaryPrimitiveKernel);
  UnaryPrimitiveKernel() = default;
  ~UnaryPrimitiveKernel() = default;

  using PrimitiveFactoryFuncType = std::function<std::unique_ptr<ep::primitive::ElementwiseUnary>(
      user_op::KernelComputeContext*)>;

  UnaryPrimitiveKernel(const std::string& output_name, const std::string& input_name,
                       PrimitiveFactoryFuncType fn)
      : output_name_(output_name),
        input_name_(input_name),
        primitive_factory_func_(std::move(fn)) {}

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    auto primitive = primitive_factory_func_(ctx);
    CHECK(primitive);

    const user_op::Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex(input_name_, 0);
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex(output_name_, 0);

    const ShapeView& input_shape = input_tensor->shape_view();
    const ShapeView& output_shape = output_tensor->shape_view();
    CHECK_EQ(input_shape, output_shape) << "Input shape should be equal to Output shape.";
    const int64_t elem_cnt = input_shape.elem_cnt();

    if (elem_cnt != 0) {
      primitive->Launch(ctx->stream(), input_tensor->dptr(), output_tensor->mut_dptr(), elem_cnt);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::string output_name_;
  std::string input_name_;
  PrimitiveFactoryFuncType primitive_factory_func_;
};

class BinaryPrimitiveKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryPrimitiveKernel);
  BinaryPrimitiveKernel() = default;
  ~BinaryPrimitiveKernel() = default;

  using PrimitiveFactoryFuncType =
      std::function<std::unique_ptr<ep::primitive::BroadcastElementwiseBinary>(
          user_op::KernelComputeContext*)>;

  BinaryPrimitiveKernel(const std::string& output_name, const std::string& input_a_name,
                        const std::string& input_b_name, PrimitiveFactoryFuncType fn)
      : output_name_(output_name),
        input_a_name_(input_a_name),
        input_b_name_(input_b_name),
        primitive_factory_func_(std::move(fn)) {}

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    auto primitive = primitive_factory_func_(ctx);
    CHECK(primitive);

    const user_op::Tensor* input_a_tensor = ctx->Tensor4ArgNameAndIndex(input_a_name_, 0);
    const user_op::Tensor* input_b_tensor = ctx->Tensor4ArgNameAndIndex(input_b_name_, 0);
    user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex(output_name_, 0);

    const ShapeView& input_a_shape = input_a_tensor->shape_view();
    const ShapeView& input_b_shape = input_b_tensor->shape_view();
    const ShapeView& output_shape = output_tensor->shape_view();
    CHECK_EQ(input_a_shape, input_b_shape) << "InputA shape should be equal to InputB shape.";
    CHECK_EQ(input_a_shape, output_shape) << "Input shape should be equal to Output shape.";
    const int64_t elem_cnt = input_a_shape.elem_cnt();

    if (elem_cnt != 0) {
      primitive->Launch(ctx->stream(), input_a_shape.NumAxes(), input_a_shape.ptr(),
                        input_a_tensor->dptr(), input_b_shape.NumAxes(), input_b_shape.ptr(),
                        input_b_tensor->dptr(), output_tensor->mut_dptr());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::string output_name_;
  std::string input_a_name_;
  std::string input_b_name_;
  PrimitiveFactoryFuncType primitive_factory_func_;
};

namespace {
auto UnaryPrimitiveExists(ep::primitive::UnaryOp op, const std::string& output_name,
                          const std::string& input_name) {
  return hob::make_custom(
      "ElementwiseUnaryPrimitiveExists", [=](const user_op::KernelRegContext& ctx) {
        const user_op::TensorDesc* src = ctx.TensorDesc4ArgNameAndIndex(input_name, 0);
        const user_op::TensorDesc* dst = ctx.TensorDesc4ArgNameAndIndex(output_name, 0);
        auto primitive = ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
            ctx.device_type(), op, src->data_type(), dst->data_type());
        return primitive.operator bool();
      });
}

auto BinaryPrimitiveExists(ep::primitive::BinaryOp op, const std::string& output_name,
                           const std::string& input_a_name) {
  return hob::make_custom(
      "BroadcastElementwiseBinaryPrimitiveExists", [=](const user_op::KernelRegContext& ctx) {
        const user_op::TensorDesc* src0 = ctx.TensorDesc4ArgNameAndIndex(input_a_name, 0);
        const user_op::TensorDesc* dst = ctx.TensorDesc4ArgNameAndIndex(output_name, 0);
        auto primitive =
            ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx.device_type(), op, src0->data_type(), dst->data_type(), 1 /*max_num_dims*/);
        return primitive.operator bool();
      });
}
}  // namespace

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ELEMENTWISE_XPU_KERNEL_H_
