#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

std::string AddReshapeLikeOp(const std::string& op_name, const std::string& in_lbn,
                             const std::string& like_lbn, user_op::AddOpFn AddOp) {
  user_op::UserOpConfWrapperBuilder builder(op_name);
  user_op::UserOpConfWrapper grad_op =
      builder.Op("reshape_like").Input("in", in_lbn).Input("like", like_lbn).Output("out").Build();
  AddOp(grad_op);
  return grad_op.output("out", 0);
}

std::string AddReduceSumLikeOp(const std::string& op_name, const std::string& in_lbn,
                               const std::string& like_lbn, const AxisVector& broadcast_axis_vec,
                               user_op::AddOpFn AddOp) {
  user_op::UserOpConfWrapperBuilder builder(op_name);
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

void GenBackwardOpConf4BroadcastAdd(const std::string& input_arg_name,
                                    const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  const user_op::TensorDesc& z_desc = op.TensorDesc4ArgNameAndIndex("z", 0);
  const user_op::TensorDesc& in_desc = op.TensorDesc4ArgNameAndIndex(input_arg_name, 0);
  const Shape& left_extended_shape =
      CreateLeftExtendedShape(ShapeView(in_desc.shape()), z_desc.shape().NumAxes());
  if (in_desc.shape() == z_desc.shape()) {
    op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("z", 0), input_arg_name, 0);
  } else if (left_extended_shape == z_desc.shape()) {
    const std::string& reshape_like_out_lbn =
        AddReshapeLikeOp(op.op_name() + "_grad_" + input_arg_name + "_reshape_like",
                         op.GetGradTensorWithOpOutput("z", 0), op.input(input_arg_name, 0), AddOp);
    op.BindGradTensorWithOpInput(reshape_like_out_lbn, input_arg_name, 0);
  } else {
    const AxisVector& broadcast_axis_vec = left_extended_shape.Axes4BroadcastTo(z_desc.shape());
    const std::string& reduce_sum_like_out_lbn =
        AddReduceSumLikeOp(op.op_name() + "_grad_" + input_arg_name + "_reduce_sum_like",
                           op.GetGradTensorWithOpOutput("z", 0), op.input(input_arg_name, 0),
                           broadcast_axis_vec, AddOp);
    op.BindGradTensorWithOpInput(reduce_sum_like_out_lbn, input_arg_name, 0);
  }
}

void GenBackwardOpConf4BroadcastMul(const std::string& input_x_arg_name,
                                    const std::string& input_y_arg_name,
                                    const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  const user_op::TensorDesc& z_desc = op.TensorDesc4ArgNameAndIndex("z", 0);
  const user_op::TensorDesc& in_desc = op.TensorDesc4ArgNameAndIndex(input_x_arg_name, 0);
  user_op::UserOpConfWrapperBuilder broadcast_mul_builder(op.op_name() + "_grad_" + input_x_arg_name
                                                          + "_mul");
  user_op::UserOpConfWrapper broadcast_mul_op =
      broadcast_mul_builder.Op("broadcast_mul")
          .Input("x", op.GetGradTensorWithOpOutput("z", 0))
          .Input("y", op.input(input_y_arg_name, 0))
          .Output("z")
          .Build();
  AddOp(broadcast_mul_op);

  const Shape& left_extended_shape =
      CreateLeftExtendedShape(ShapeView(in_desc.shape()), z_desc.shape().NumAxes());

  if (in_desc.shape() == z_desc.shape()) {
    op.BindGradTensorWithOpInput(broadcast_mul_op.output("z", 0), input_x_arg_name, 0);
  } else if (left_extended_shape == z_desc.shape()) {
    const std::string& reshape_like_out_lbn =
        AddReshapeLikeOp(op.op_name() + "_grad_" + input_x_arg_name + "_reshape_like",
                         broadcast_mul_op.output("z", 0), op.input(input_x_arg_name, 0), AddOp);
    op.BindGradTensorWithOpInput(reshape_like_out_lbn, input_x_arg_name, 0);

  } else {
    const AxisVector& broadcast_axis_vec = left_extended_shape.Axes4BroadcastTo(z_desc.shape());
    const std::string& reduce_sum_like_out_lbn = AddReduceSumLikeOp(
        op.op_name() + "_grad_" + input_x_arg_name + "_reduce_sum_like",
        broadcast_mul_op.output("z", 0), op.input(input_x_arg_name, 0), broadcast_axis_vec, AddOp);
    op.BindGradTensorWithOpInput(reduce_sum_like_out_lbn, input_x_arg_name, 0);
  }
}

}  // namespace

REGISTER_USER_OP_GRAD("broadcast_add")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) { GenBackwardOpConf4BroadcastAdd("x", op, AddOp); }
      if (op.NeedGenGradTensor4OpInput("y", 0)) { GenBackwardOpConf4BroadcastAdd("y", op, AddOp); }
    });

REGISTER_USER_OP_GRAD("broadcast_sub")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      const user_op::TensorDesc& z_desc = op.TensorDesc4ArgNameAndIndex("z", 0);
      if (op.NeedGenGradTensor4OpInput("x", 0)) { GenBackwardOpConf4BroadcastAdd("x", op, AddOp); }
      if (op.NeedGenGradTensor4OpInput("y", 0)) {
        user_op::UserOpConfWrapperBuilder scalar_mul_builder(op.op_name() + "_grad_y_mul");
        user_op::UserOpConfWrapper scalar_mul_op =
            scalar_mul_builder.Op("scalar_mul")
                .Input("in", op.GetGradTensorWithOpOutput("z", 0))
                .Attr("has_int_operand", false)
                .Attr("has_float_operand", true)
                .Attr("float_operand", -1)
                .Output("out")
                .Build();
        AddOp(scalar_mul_op);

        const user_op::TensorDesc& y_desc = op.TensorDesc4ArgNameAndIndex("y", 0);
        const Shape& left_extended_shape =
            CreateLeftExtendedShape(ShapeView(y_desc.shape()), z_desc.shape().NumAxes());

        if (y_desc.shape() == z_desc.shape()) {
          op.BindGradTensorWithOpInput(scalar_mul_op.output("out", 0), "y", 0);
        } else if (left_extended_shape == z_desc.shape()) {
          const std::string& reshape_like_out_lbn =
              AddReshapeLikeOp(op.op_name() + "_grad_y_reshape_like",
                               scalar_mul_op.output("out", 0), op.input("y", 0), AddOp);
          op.BindGradTensorWithOpInput(reshape_like_out_lbn, "y", 0);
        } else {
          const AxisVector& broadcast_axis_vec =
              left_extended_shape.Axes4BroadcastTo(z_desc.shape());
          const std::string& reduce_sum_like_out_lbn = AddReduceSumLikeOp(
              op.op_name() + "_grad_y_reduce_sum_like", scalar_mul_op.output("out", 0),
              op.input("y", 0), broadcast_axis_vec, AddOp);
          op.BindGradTensorWithOpInput(reduce_sum_like_out_lbn, "y", 0);
        }
      }
    });

REGISTER_USER_OP_GRAD("broadcast_mul")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        GenBackwardOpConf4BroadcastMul("x", "y", op, AddOp);
      }
      if (op.NeedGenGradTensor4OpInput("y", 0)) {
        GenBackwardOpConf4BroadcastMul("y", "x", op, AddOp);
      }
    });

REGISTER_USER_OP_GRAD("broadcast_div")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        const user_op::TensorDesc& z_desc = op.TensorDesc4ArgNameAndIndex("z", 0);
        const user_op::TensorDesc& x_desc = op.TensorDesc4ArgNameAndIndex("x", 0);
        user_op::UserOpConfWrapperBuilder broadcast_div_builder(op.op_name() + "_grad_x_div");
        user_op::UserOpConfWrapper broadcast_div_op =
            broadcast_div_builder.Op("broadcast_div")
                .Input("x", op.GetGradTensorWithOpOutput("z", 0))
                .Input("y", op.input("y", 0))
                .Output("z")
                .Build();
        AddOp(broadcast_div_op);

        const Shape& left_extended_shape =
            CreateLeftExtendedShape(ShapeView(x_desc.shape()), z_desc.shape().NumAxes());

        if (x_desc.shape() == z_desc.shape()) {
          op.BindGradTensorWithOpInput(broadcast_div_op.output("z", 0), "x", 0);
        } else if (left_extended_shape == z_desc.shape()) {
          const std::string& reshape_like_out_lbn =
              AddReshapeLikeOp(op.op_name() + "_grad_x_reshape_like",
                               broadcast_div_op.output("z", 0), op.input("x", 0), AddOp);
          op.BindGradTensorWithOpInput(reshape_like_out_lbn, "x", 0);

        } else {
          const AxisVector& broadcast_axis_vec =
              left_extended_shape.Axes4BroadcastTo(z_desc.shape());
          const std::string& reduce_sum_like_out_lbn = AddReduceSumLikeOp(
              op.op_name() + "_grad_x_reduce_sum_like", broadcast_div_op.output("z", 0),
              op.input("x", 0), broadcast_axis_vec, AddOp);
          op.BindGradTensorWithOpInput(reduce_sum_like_out_lbn, "x", 0);
        }
      }
      if (op.NeedGenGradTensor4OpInput("y", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_y_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("broadcast_div_grad")
                                                 .Input("y", op.input("y", 0))
                                                 .Input("z", op.output("z", 0))
                                                 .Input("dz", op.GetGradTensorWithOpOutput("z", 0))
                                                 .Output("dy")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dy", 0), "y", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
