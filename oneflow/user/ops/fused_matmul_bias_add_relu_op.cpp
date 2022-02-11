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
#include "oneflow/core/common/just.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc4FusedMatmul(user_op::InferContext* ctx) {
  // todo: add bias add check.

  // bool transpose_a = ctx->Attr<bool>("transpose_a"); // false
  // bool transpose_b = ctx->Attr<bool>("transpose_b"); // true

  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  const user_op::TensorDesc& bias1 = ctx->InputTensorDesc("bias1", 0);
  const user_op::TensorDesc& c = ctx->InputTensorDesc("c", 0);
  const user_op::TensorDesc& bias2 = ctx->InputTensorDesc("bias2", 0);

  CHECK_EQ_OR_RETURN(a.shape().NumAxes(), b.shape().NumAxes())
      << "Num axes size of a and b should be equal.";
  CHECK_EQ_OR_RETURN(b.shape().NumAxes(), c.shape().NumAxes())
      << "Num axes size of b and c should be equal.";

  CHECK_EQ_OR_RETURN(bias1.shape().NumAxes(), 1) << "Bias1 num axes size should be 1.";
  CHECK_EQ_OR_RETURN(bias2.shape().NumAxes(), 1) << "Bias2 num axes size should be 1.";
  // Currently only support 2d matmul. 
  CHECK_EQ_OR_RETURN(a.shape().NumAxes(), 2);
  size_t num_axes = a.shape().NumAxes();

  user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);

  *ctx->OutputShape("out", 0) = ctx->InputShape("a", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("a", 0);

  // （m, k) 
  //  (n, k) need transpose
  //  (j, n)need transpose
  int64_t m = 0, n = 0, k = 0, j = 0;  // tensor a (no trans): m*k, tensor b (no trans): k*n
  m = a.shape().At(num_axes - 2);
  k = a.shape().At(num_axes - 1);
  CHECK_EQ_OR_RETURN(k, b.shape().At(num_axes - 1));
  n = b.shape().At(num_axes - 2);
  CHECK_EQ_OR_RETURN(bias1.shape().At(0), n)
      << "Bias1 shape cannot be added (" << bias1.shape().At(0) << ") and (" << n << ")";
  CHECK_EQ_OR_RETURN(n, c.shape().At(num_axes - 1));
  j = c.shape().At(num_axes - 2);
  CHECK_EQ_OR_RETURN(bias2.shape().At(0), j)  
      << "Bias2 shape cannot be added (" << bias2.shape().At(0) << ") and (" << j << ")";

  out->mut_shape()->Set(num_axes - 2, m);
  out->mut_shape()->Set(num_axes - 1, j);
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4Matmul(user_op::InferContext* ctx) {
  const DataType& dtype = ctx->InputDType("a", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), dtype);
  CHECK_EQ_OR_RETURN(ctx->InputDType("bias1", 0), dtype);
  CHECK_EQ_OR_RETURN(ctx->InputDType("c", 0), dtype);
  CHECK_EQ_OR_RETURN(ctx->InputDType("bias2", 0), dtype);
  *ctx->OutputDType("out", 0) = dtype;
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> FusedMatmulBiasAddReluOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4FusedMatmul(ctx);
}

/*static*/ Maybe<void> FusedMatmulBiasAddReluOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedMatmulBiasAddReluOp::GetSbp(user_op::SbpContext* ctx) {
  // (m, k_a) * (k_b, n) where k_a == k_b
  // TODO
  // int32_t m_axis = -1;
  // int32_t n_axis = -1;
  // if (ctx->Attr<bool>("transpose_a")) {
  //   m_axis = 1;
  // } else {
  //   m_axis = 0;
  // }
  // if (ctx->Attr<bool>("transpose_b")) {
  //   n_axis = 0;
  // } else {
  //   n_axis = 1;
  // }
  // ctx->NewBuilder()
  //     .Split(user_op::OpArg("a", 0), m_axis)
  //     .Broadcast(user_op::OpArg("b", 0))
  //     .Broadcast(user_op::OpArg("bias", 0))
  //     .Split(user_op::OpArg("out", 0), 0)
  //     .Build();
  // ctx->NewBuilder()
  //     .Broadcast(user_op::OpArg("a", 0))
  //     .Split(user_op::OpArg("b", 0), n_axis)
  //     .Split(user_op::OpArg("bias", 0), 0)
  //     .Split(user_op::OpArg("out", 0), 1)
  //     .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedMatmulBiasAddReluOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Matmul(ctx);
}

// REGISTER_USER_OP_GRAD("fused_matmul_bias_add_relu")
//     .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
//                                const user_op::AddOpFn& AddOp) -> Maybe<void> {
//       if (op.NeedGenGradTensor4OpInput("a", 0) || op.NeedGenGradTensor4OpInput("b", 0)
//           || op.NeedGenGradTensor4OpInput("bias", 0)) {
//         double alpha = op.attr<double>("alpha");
//         bool transpose_a = op.attr<bool>("transpose_a");
//         bool transpose_b = op.attr<bool>("transpose_b");
//         user_op::UserOpConfWrapperBuilder relu_grad_builder(op.op_name() + "_relu_grad");
//         user_op::UserOpConfWrapper relu_grad_op =
//             relu_grad_builder.Op("relu_grad")
//                 .Input("y", op.output("out", 0))
//                 .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
//                 .Output("dx")
//                 .Build();
//         AddOp(relu_grad_op);
//         if (op.NeedGenGradTensor4OpInput("bias", 0)) {
//           // TODO: Currently Only support 2d fused_matmul.
//           // so here we hard encode bias reduce axis as 0.
//           std::vector<int32_t> reduce_axes_vec{0};
//           user_op::UserOpConfWrapperBuilder bias_grad_builder(op.op_name() + "_bias_grad");
//           user_op::UserOpConfWrapper bias_grad_op =
//               bias_grad_builder.Op("reduce_sum")
//                   .Input("input_tensor", relu_grad_op.output("dx", 0))
//                   .Output("output_tensor")
//                   .Attr("axis", reduce_axes_vec)
//                   .Attr("keepdims", false)
//                   .Build();
//           AddOp(bias_grad_op);
//           op.BindGradTensorWithOpInput(bias_grad_op.output("output_tensor", 0), "bias", 0);
//         }
//         if (op.NeedGenGradTensor4OpInput("a", 0)) {
//           user_op::UserOpConfWrapperBuilder matmul_a_grad_builder(op.op_name()
//                                                                   + "_matmul_a_grad");  // todo
//           if (transpose_a) {
//             user_op::UserOpConfWrapper matmul_a_grad_op =
//                 matmul_a_grad_builder.Op("matmul")
//                     .Input("a", op.input("b", 0))
//                     .Input("b", relu_grad_op.output("dx", 0))
//                     .Output("out")
//                     .Attr<bool>("transpose_a", transpose_b)
//                     .Attr<bool>("transpose_b", true)
//                     .Attr<double>("alpha", alpha)
//                     .Build();
//             AddOp(matmul_a_grad_op);
//             op.BindGradTensorWithOpInput(matmul_a_grad_op.output("out", 0), "a", 0);
//           } else {
//             user_op::UserOpConfWrapper matmul_a_grad_op =
//                 matmul_a_grad_builder.Op("matmul")
//                     .Input("a", relu_grad_op.output("dx", 0))
//                     .Input("b", op.input("b", 0))
//                     .Output("out")
//                     .Attr<bool>("transpose_a", false)
//                     .Attr<bool>("transpose_b", !transpose_b)
//                     .Attr<double>("alpha", alpha)
//                     .Build();
//             AddOp(matmul_a_grad_op);
//             op.BindGradTensorWithOpInput(matmul_a_grad_op.output("out", 0), "a", 0);
//           }
//         }
//         if (op.NeedGenGradTensor4OpInput("b", 0)) {
//           user_op::UserOpConfWrapperBuilder matmul_b_grad_builder(op.op_name()
//                                                                   + "_matmul_b_grad");  // todo
//           if (transpose_b) {
//             user_op::UserOpConfWrapper matmul_b_grad_op =
//                 matmul_b_grad_builder.Op("matmul")
//                     .Input("a", relu_grad_op.output("dx", 0))
//                     .Input("b", op.input("a", 0))
//                     .Output("out")
//                     .Attr<bool>("transpose_a", true)
//                     .Attr<bool>("transpose_b", transpose_a)
//                     .Attr<double>("alpha", alpha)
//                     .Build();
//             AddOp(matmul_b_grad_op);
//             op.BindGradTensorWithOpInput(matmul_b_grad_op.output("out", 0), "b", 0);
//           } else {
//             user_op::UserOpConfWrapper matmul_b_grad_op =
//                 matmul_b_grad_builder.Op("matmul")
//                     .Input("a", op.input("a", 0))
//                     .Input("b", relu_grad_op.output("dx", 0))
//                     .Output("out")
//                     .Attr<bool>("transpose_a", !transpose_a)
//                     .Attr<bool>("transpose_b", false)
//                     .Attr<double>("alpha", alpha)
//                     .Build();
//             AddOp(matmul_b_grad_op);
//             op.BindGradTensorWithOpInput(matmul_b_grad_op.output("out", 0), "b", 0);
//           }
//         }
//       }
//       return Maybe<void>::Ok();
//     });

}  // namespace oneflow
