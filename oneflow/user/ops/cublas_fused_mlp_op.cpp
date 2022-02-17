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
// #if CUDA_VERSION >= 11040
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc4FusedMatmul(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);
  int32_t weight_size = ctx->input_size("weights"); 
  int32_t bias_size = ctx->input_size("biases"); 
  CHECK_EQ_OR_RETURN(weight_size, bias_size); 
  /*
  A: (m, k)
  B: (n, k) need transpose
  C: (m, n)
  */
  int64_t m = 0, n = 0, k = 0;
  m = x_desc.shape().At(0); 
  k = x_desc.shape().At(1); 

  for (int32_t idx=0; idx < weight_size; idx++) {
    // skip first input weight. 
    const user_op::TensorDesc& weight_desc =
        ctx->InputTensorDesc("weights", idx);
    const user_op::TensorDesc& bias_desc =
        ctx->InputTensorDesc("biases", idx);
    CHECK_EQ_OR_RETURN(weight_desc.shape().NumAxes(), 2); 
    CHECK_EQ_OR_RETURN(bias_desc.shape().NumAxes(), 1); 

    n = weight_desc.shape().At(0); 
    CHECK_EQ_OR_RETURN(bias_desc.shape().At(0), n); 
    CHECK_EQ_OR_RETURN(weight_desc.shape().At(1), k); 
    
    // Set for next layer. 
    k = n; 
    // Set Middle result shape. 
    user_op::TensorDesc* cublas_aux_desc = ctx->OutputTensorDesc("cublas_aux", idx);
    *ctx->OutputShape("cublas_aux", idx) = x_desc.shape();
    cublas_aux_desc->mut_shape()->Set(1, k);
    user_op::TensorDesc* hidden_desc = ctx->OutputTensorDesc("hidden", idx);
    *ctx->OutputShape("hidden", idx) = x_desc.shape();
    hidden_desc->mut_shape()->Set(1, k);
  }
  *ctx->OutputShape("out", 0) = x_desc.shape();
  out->mut_shape()->Set(1, n);
  return Maybe<void>::Ok();
}


Maybe<void> InferDataType4Matmul(user_op::InferContext* ctx){
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("x", 0);
  
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.data_type(), first_in_desc.data_type());
  }

  for (const auto& out_arg_pair : ctx->outputs()) {
    user_op::TensorDesc* out_desc =
        ctx->OutputTensorDesc(out_arg_pair.first, out_arg_pair.second);
    *out_desc->mut_data_type() = first_in_desc.data_type();
  }
  return Maybe<void>::Ok(); 
}


}  // namespace

/* static */ Maybe<void> CublasFusedMLPOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4FusedMatmul(ctx);
}

/*static*/ Maybe<void> CublasFusedMLPOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CublasFusedMLPOp::GetSbp(user_op::SbpContext* ctx) {
  // Currently Only support S0 B B B B ... S0
  auto builder = ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0); 
  for(int i =0; i<ctx->user_op_conf().input_size("weights"); ++i) {
    builder.Broadcast(user_op::OpArg("weights", i)); 
  }
  for(int i =0; i<ctx->user_op_conf().input_size("biases"); ++i) {
    builder.Broadcast(user_op::OpArg("biases", i));
  }
  for(int i =0; i<ctx->user_op_conf().output_size("cublas_aux"); ++i) {
    builder.Split(user_op::OpArg("cublas_aux", i), 0);
  }
  for(int i =0; i<ctx->user_op_conf().output_size("hidden"); ++i) {
    builder.Split(user_op::OpArg("hidden", i), 0);
  }
  builder.Split(user_op::OpArg("out", 0), 0); 
  builder.Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CublasFusedMLPOp::InferDataType(user_op::InferContext* ctx) {
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

// #endif // CUDA_VERSION >= 11040
