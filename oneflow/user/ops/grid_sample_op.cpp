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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

Maybe<void> GridSampleOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                    const user_op::UserOpConfWrapper& conf) {
  bool pass_checked = true;
  std::stringstream err;
  err << "Illegal value for " << conf.op_type_name() << " op " << conf.op_name() << ": ";

  const auto& interpolation_mode = conf.attr<std::string>("interpolation_mode");
  if (!(interpolation_mode == "bilinear" || interpolation_mode == "nearest"
        || interpolation_mode == "bicubic")) {
    err << " interpolation_mode:" << interpolation_mode;
    pass_checked = false;
  }

  const auto& padding_mode = conf.attr<std::string>("padding_mode");
  if (!(padding_mode == "zeros" || padding_mode == "border" || padding_mode == "reflection")) {
    err << " padding_mode:" << padding_mode;
    pass_checked = false;
  }

  if (pass_checked) {
    return Maybe<void>::Ok();
  } else {
    return oneflow::Error::CheckFailedError() << err.str();
  }
}

/*static*/ auto GridSampleOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& input = ctx->InputTensorDesc("input", 0);
  const user_op::TensorDesc& grid = ctx->InputTensorDesc("grid", 0);
  user_op::TensorDesc& output = *(ctx->MutOutputTensorDesc("output", 0));
  // Only support 4D or 5D input with NCHW layout
  // For 4D grid: input  = { N, C, H_in, W_in },
  //              grid   = { N, H_out, W_out, 2 }
  //              output = { N, C, H_out, W_out }
  // For 5D grid: input  = { N, C, D_in, H_in, W_in },
  //              grid   = { N, D_out, H_out, W_out, 3 }
  //              output = { N, C, D_out, H_out, W_out }
  const Shape& input_shape = input.shape();
  const Shape& grid_shape = grid.shape();

  bool is_4d_input = true;
  if (input_shape.NumAxes() == 4) {
    CHECK_EQ_OR_RETURN(grid_shape.NumAxes(), 4) << "Grid and input MUST have same dimention";
    CHECK_EQ_OR_RETURN(grid_shape.At(3), 2) << "Grid shape MUST (N, H_out, W_out, 2)";
    is_4d_input = true;
  } else if (input_shape.NumAxes() == 5) {
    CHECK_EQ_OR_RETURN(grid_shape.NumAxes(), 5) << "Grid and input MUST have same dimention";
    CHECK_EQ_OR_RETURN(grid_shape.At(4), 3) << "Grid shape MUST (N, H_out, W_out, 3)";
    if (ctx->Attr<std::string>("interpolation_mode") == "bicubic") {
      oneflow::Error::CheckFailedError() << "Mode='bicubic' supports only 4-D input";
    }
    is_4d_input = false;
  } else {
    CHECK_OR_RETURN(false) << "MUST be 4D or 5D input";
  }
  output.set_is_dynamic(grid.is_dynamic());
  if (is_4d_input) {
    output.set_shape(
        Shape({input_shape.At(0), input_shape.At(1), grid_shape.At(1), grid_shape.At(2)}));
  } else {
    output.set_shape(Shape({input_shape.At(0), input_shape.At(1), grid_shape.At(1),
                            grid_shape.At(2), grid_shape.At(3)}));
  }
  return Maybe<void>::Ok();
}
/*static*/ auto GridSampleOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return GridSampleOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto GridSampleOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("grid", 0), 0)
      .Split(user_op::OpArg("output", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("input", 0), 1)
      .Broadcast(user_op::OpArg("grid", 0))
      .Split(user_op::OpArg("output", 0), 1)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ auto GridSampleOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  ctx->SetOutputDType("output", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

Maybe<void> GridSampleGradOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                        const user_op::UserOpConfWrapper& conf) {
  return GridSampleOp::CheckAttr(def, conf);
}

/*static*/ auto GridSampleGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  ctx->MutOutputTensorDesc("dinput", 0)->set_shape(ctx->InputTensorDesc("input", 0).shape());
  ctx->MutOutputTensorDesc("dgrid", 0)->set_shape(ctx->InputTensorDesc("grid", 0).shape());
  return Maybe<void>::Ok();
}
/*static*/ auto GridSampleGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return GridSampleGradOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto GridSampleGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("doutput", 0), 0)
      .Split(user_op::OpArg("input", 0), 0)
      .Split(user_op::OpArg("grid", 0), 0)
      .Split(user_op::OpArg("dinput", 0), 0)
      .Split(user_op::OpArg("dgrid", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("doutput", 0), 1)
      .Split(user_op::OpArg("input", 0), 1)
      .Broadcast(user_op::OpArg("grid", 0))
      .Split(user_op::OpArg("dinput", 0), 1)
      .PartialSum(user_op::OpArg("dgrid", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ auto GridSampleGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  ctx->SetOutputDType("dinput", 0, ctx->InputDType("input", 0));
  ctx->SetOutputDType("dgrid", 0, ctx->InputDType("grid", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
