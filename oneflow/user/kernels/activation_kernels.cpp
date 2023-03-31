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
#include "oneflow/core/ep/include/primitive/binary_op.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/user/kernels/elementwise_primitive_kernel.h"

namespace oneflow {

REGISTER_USER_KERNEL("elu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kElu, src->data_type(),
                dst->data_type(), ctx->Attr<double>("alpha"));
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kElu, "out", "in"));

REGISTER_USER_KERNEL("elu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kEluBackwardWithDyX, src->data_type(),
                dst->data_type(), 1 /*max_num_dims*/, ctx->Attr<double>("alpha"));
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kEluBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("celu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kCelu, src->data_type(),
                dst->data_type(), ctx->Attr<double>("alpha"));
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kCelu, "out", "in"));

REGISTER_USER_KERNEL("celu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "y", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kCeluBackwardWithDyY, src->data_type(),
                dst->data_type(), 1 /*max_num_dims*/, ctx->Attr<double>("alpha"));
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kCeluBackwardWithDyY, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("hardswish")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kHardSwish, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kHardSwish, "out", "in"));

REGISTER_USER_KERNEL("hardswish_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kHardswishBackwardWithDyX,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kHardswishBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("hardsigmoid")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kHardSigmoid, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kHardSigmoid, "out", "in"));

REGISTER_USER_KERNEL("hardsigmoid_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kHardsigmoidBackwardWithDyX,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kHardsigmoidBackwardWithDyX,
                                           "dx", "dy"));

REGISTER_USER_KERNEL("hardshrink")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kHardShrink, src->data_type(),
                dst->data_type(), ctx->Attr<double>("lambd"));
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kHardShrink, "out", "in"))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("hardshrink_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "y", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kHardshrinkBackwardWithDyY,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/, ctx->Attr<double>("lambd"));
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kHardshrinkBackwardWithDyY,
                                           "dx", "dy"))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("hardtanh")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kHardTanh, src->data_type(),
                dst->data_type(), ctx->Attr<double>("min_val"), ctx->Attr<double>("max_val"));
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kHardTanh, "out", "in"))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("hardtanh_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "y", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kHardtanhBackwardWithDyY,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/,
                ctx->Attr<double>("min_val"), ctx->Attr<double>("max_val"));
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kHardtanhBackwardWithDyY, "dx",
                                           "dy"))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("gelu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kGelu, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kGelu, "out", "in"));

REGISTER_USER_KERNEL("gelu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kGeluBackwardWithDyX, src->data_type(),
                dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kGeluBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("fast_gelu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kFastGelu, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kFastGelu, "out", "in"));

REGISTER_USER_KERNEL("fast_gelu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kFastGeluBackwardWithDyX,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kFastGeluBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("quick_gelu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "y", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kQuickGelu, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kQuickGelu, "y", "x"));

REGISTER_USER_KERNEL("quick_gelu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kQuickGeluBackwardWithDyX,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kQuickGeluBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("leaky_relu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "y", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kLeakyRelu, src->data_type(),
                dst->data_type(), ctx->Attr<float>("alpha"));
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kLeakyRelu, "y", "x"));

REGISTER_USER_KERNEL("leaky_relu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kLeakyReluBackwardWithDyX,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/, ctx->Attr<float>("alpha"));
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kLeakyReluBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("mish")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kMish, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kMish, "out", "in"));

REGISTER_USER_KERNEL("mish_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kMishBackwardWithDyX, src->data_type(),
                dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kMishBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("relu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "y", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kRelu, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kRelu, "y", "x"))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("relu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "y", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kReluBackwardWithDyY, src->data_type(),
                dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kReluBackwardWithDyY, "dx",
                                           "dy"))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("silu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kSilu, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSilu, "out", "in"));

REGISTER_USER_KERNEL("silu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kSiluBackwardWithDyX, src->data_type(),
                dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kSiluBackwardWithDyX, "dx",
                                           "dy"));
REGISTER_USER_KERNEL("trunc")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kTrunc, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kTrunc, "out", "in"));

REGISTER_USER_KERNEL("selu")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kSelu, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSelu, "out", "in"));

REGISTER_USER_KERNEL("selu_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kSeluBackwardWithDyX, src->data_type(),
                dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kSeluBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("softshrink")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kSoftShrink, src->data_type(),
                dst->data_type(), ctx->Attr<double>("alpha"));
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSoftShrink, "out", "in"));

REGISTER_USER_KERNEL("softshrink_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "y", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kSoftshrinkBackwardWithDyY,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/, ctx->Attr<double>("alpha"));
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kSoftshrinkBackwardWithDyY,
                                           "dx", "dy"));

REGISTER_USER_KERNEL("softsign")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kSoftSign, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSoftSign, "out", "in"));

REGISTER_USER_KERNEL("softsign_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kSoftsignBackwardWithDyX,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kSoftsignBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("softplus")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kSoftPlus, src->data_type(),
                dst->data_type(), ctx->Attr<double>("beta"), ctx->Attr<double>("threshold"));
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kSoftPlus, "out", "in"));

REGISTER_USER_KERNEL("softplus_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kSoftplusBackwardWithDyX,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/, ctx->Attr<double>("beta"),
                ctx->Attr<double>("threshold"));
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kSoftplusBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("tanh")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "y", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("x", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("y", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kTanh, src->data_type(),
                dst->data_type());
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kTanh, "y", "x"));

REGISTER_USER_KERNEL("tanh_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kTanhBackwardWithDyX, src->data_type(),
                dst->data_type(), 1 /*max_num_dims*/);
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kTanhBackwardWithDyX, "dx",
                                           "dy"));

REGISTER_USER_KERNEL("threshold")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<UnaryPrimitiveKernel>(
          "out", "in", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("in", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
            return ep::primitive::NewPrimitive<ep::primitive::ElementwiseUnaryFactory>(
                ctx->device_type(), ep::primitive::UnaryOp::kThreshold, src->data_type(),
                dst->data_type(), ctx->Attr<double>("threshold_val"), ctx->Attr<double>("value"));
          });
    })
    .SetIsMatchedHob(UnaryPrimitiveExists(ep::primitive::UnaryOp::kThreshold, "out", "in"));

REGISTER_USER_KERNEL("threshold_grad")
    .SetCreateFn([]() {
      return user_op::NewOpKernel<BinaryPrimitiveKernel>(
          "dx", "dy", "x", [](user_op::KernelComputeContext* ctx) {
            const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
            const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
            return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
                ctx->device_type(), ep::primitive::BinaryOp::kThresholdBackwardWithDyX,
                src->data_type(), dst->data_type(), 1 /*max_num_dims*/,
                ctx->Attr<double>("threshold_val"));
          });
    })
    .SetIsMatchedHob(BinaryPrimitiveExists(ep::primitive::BinaryOp::kThresholdBackwardWithDyX, "dx",
                                           "dy"));

}  // namespace oneflow
