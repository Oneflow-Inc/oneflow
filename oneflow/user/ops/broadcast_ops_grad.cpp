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

namespace oneflow {

namespace {

std::string CreateReduceSumLikeBlob(const std::string& in_lbn, const Shape& in_shape,
                                    const std::string& like_lbn, const Shape& like_shape,
                                    const std::string& op_name, user_op::AddOpFn AddOp) {
  const Shape& left_extended_shape =
      CreateLeftExtendedShape(ShapeView(like_shape), in_shape.NumAxes());
  if (in_shape == like_shape) {
    return in_lbn;
  } else if (in_shape == left_extended_shape) {
    user_op::UserOpConfWrapperBuilder builder(op_name + "_grad_reshape_like");
    user_op::UserOpConfWrapper grad_op = builder.Op("reshape_like")
                                             .Input("in", in_lbn)
                                             .Input("like", like_lbn)
                                             .Output("out")
                                             .Build();
    AddOp(grad_op);
    return grad_op.output("out", 0);
  } else {
    const AxisVector& broadcast_axis_vec = left_extended_shape.Axes4BroadcastTo(in_shape);
    user_op::UserOpConfWrapperBuilder builder(op_name + "_grad_reduce_sum_like");
    user_op::UserOpConfWrapper grad_op =
        builder.Op("reduce_sum_like")
            .Input("x", in_lbn)
            .Input("like", like_lbn)
            .Attr<std::vector<int32_t>>("axis",
                                        {broadcast_axis_vec.begin(), broadcast_axis_vec.end()})
            .Output("y")
            .Build();
    AddOp(grad_op);
    return grad_op.output("y", 0);
  }
}

}  // namespace

REGISTER_USER_OP_GRAD("broadcast_add")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      const Shape& z_shape = op.TensorDesc4ArgNameAndIndex("z", 0).shape();
      const std::string& dz_lbn = op.GetGradTensorWithOpOutput("z", 0);
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        const Shape& x_shape = op.TensorDesc4ArgNameAndIndex("x", 0).shape();
        const std::string& x_lbn = op.input("x", 0);
        const std::string& out_lbn =
            CreateReduceSumLikeBlob(dz_lbn, z_shape, x_lbn, x_shape, op.op_name() + "_x", AddOp);
        op.BindGradTensorWithOpInput(out_lbn, "x", 0);
      }
      if (op.NeedGenGradTensor4OpInput("y", 0)) {
        const Shape& y_shape = op.TensorDesc4ArgNameAndIndex("y", 0).shape();
        const std::string& y_lbn = op.input("y", 0);
        const std::string& out_lbn =
            CreateReduceSumLikeBlob(dz_lbn, z_shape, y_lbn, y_shape, op.op_name() + "_y", AddOp);
        op.BindGradTensorWithOpInput(out_lbn, "y", 0);
      }
    });

REGISTER_USER_OP_GRAD("broadcast_sub")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      const Shape& z_shape = op.TensorDesc4ArgNameAndIndex("z", 0).shape();
      const std::string& dz_lbn = op.GetGradTensorWithOpOutput("z", 0);
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        const Shape& x_shape = op.TensorDesc4ArgNameAndIndex("x", 0).shape();
        const std::string& x_lbn = op.input("x", 0);
        const std::string& out_lbn =
            CreateReduceSumLikeBlob(dz_lbn, z_shape, x_lbn, x_shape, op.op_name() + "_x", AddOp);
        op.BindGradTensorWithOpInput(out_lbn, "x", 0);
      }
      if (op.NeedGenGradTensor4OpInput("y", 0)) {
        user_op::UserOpConfWrapperBuilder scalar_mul_builder(op.op_name() + "_grad_y_mul");
        user_op::UserOpConfWrapper scalar_mul_op = scalar_mul_builder.Op("scalar_mul")
                                                       .Input("in", dz_lbn)
                                                       .Attr("has_int_operand", false)
                                                       .Attr("has_float_operand", true)
                                                       .Attr<int64_t>("int_operand", -1)
                                                       .Attr<double>("float_operand", -1.0)
                                                       .Output("out")
                                                       .Build();
        AddOp(scalar_mul_op);

        const Shape& y_shape = op.TensorDesc4ArgNameAndIndex("y", 0).shape();
        const std::string& y_lbn = op.input("y", 0);
        const std::string& out_lbn = CreateReduceSumLikeBlob(
            scalar_mul_op.output("out", 0), z_shape, y_lbn, y_shape, op.op_name() + "_y", AddOp);
        op.BindGradTensorWithOpInput(out_lbn, "y", 0);
      }
    });

REGISTER_USER_OP_GRAD("broadcast_mul")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      const Shape& z_shape = op.TensorDesc4ArgNameAndIndex("z", 0).shape();
      const std::string& dz_lbn = op.GetGradTensorWithOpOutput("z", 0);
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder broadcast_mul_builder(op.op_name() + "_grad_x_mul");
        user_op::UserOpConfWrapper broadcast_mul_op = broadcast_mul_builder.Op("broadcast_mul")
                                                          .Input("x", dz_lbn)
                                                          .Input("y", op.input("y", 0))
                                                          .Output("z")
                                                          .Build();
        AddOp(broadcast_mul_op);
        const Shape& x_shape = op.TensorDesc4ArgNameAndIndex("x", 0).shape();
        const std::string& x_lbn = op.input("x", 0);
        const std::string& out_lbn = CreateReduceSumLikeBlob(
            broadcast_mul_op.output("z", 0), z_shape, x_lbn, x_shape, op.op_name() + "_x", AddOp);
        op.BindGradTensorWithOpInput(out_lbn, "x", 0);
      }
      if (op.NeedGenGradTensor4OpInput("y", 0)) {
        user_op::UserOpConfWrapperBuilder broadcast_mul_builder(op.op_name() + "_grad_y_mul");
        user_op::UserOpConfWrapper broadcast_mul_op = broadcast_mul_builder.Op("broadcast_mul")
                                                          .Input("x", dz_lbn)
                                                          .Input("y", op.input("x", 0))
                                                          .Output("z")
                                                          .Build();
        AddOp(broadcast_mul_op);
        const Shape& y_shape = op.TensorDesc4ArgNameAndIndex("y", 0).shape();
        const std::string& y_lbn = op.input("y", 0);
        const std::string& out_lbn = CreateReduceSumLikeBlob(
            broadcast_mul_op.output("z", 0), z_shape, y_lbn, y_shape, op.op_name() + "_y", AddOp);
        op.BindGradTensorWithOpInput(out_lbn, "y", 0);
      }
    });

REGISTER_USER_OP_GRAD("broadcast_div")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      const std::string& dz_lbn = op.GetGradTensorWithOpOutput("z", 0);
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        const Shape& z_shape = op.TensorDesc4ArgNameAndIndex("z", 0).shape();
        user_op::UserOpConfWrapperBuilder broadcast_div_builder(op.op_name() + "_grad_x_div");
        user_op::UserOpConfWrapper broadcast_div_op = broadcast_div_builder.Op("broadcast_div")
                                                          .Input("x", dz_lbn)
                                                          .Input("y", op.input("y", 0))
                                                          .Output("z")
                                                          .Build();
        AddOp(broadcast_div_op);
        const Shape& x_shape = op.TensorDesc4ArgNameAndIndex("x", 0).shape();
        const std::string& x_lbn = op.input("x", 0);
        const std::string& out_lbn = CreateReduceSumLikeBlob(
            broadcast_div_op.output("z", 0), z_shape, x_lbn, x_shape, op.op_name() + "_x", AddOp);
        op.BindGradTensorWithOpInput(out_lbn, "x", 0);
      }
      if (op.NeedGenGradTensor4OpInput("y", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_y_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("broadcast_div_grad")
                                                 .Input("y", op.input("y", 0))
                                                 .Input("z", op.output("z", 0))
                                                 .Input("dz", dz_lbn)
                                                 .Output("dy")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dy", 0), "y", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
