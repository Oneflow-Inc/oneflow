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

Maybe<void> CheckAttr(const user_op::UserOpDefWrapper& def,
                      const user_op::UserOpConfWrapper& conf) {
  bool pass_checked = true;
  std::stringstream err;
  err << "Illegal value for " << conf.op_type_name() << " op " << conf.op_name() << ": ";

  const auto& size = conf.attr<Shape>("size");
  if (size.NumAxes() != 4 && size.NumAxes() != 5) {
    err << "dimension of size can't be:" << size.NumAxes();
    pass_checked = false;
  }

  for (int i = 0; i < size.NumAxes(); i++) {
    if (size.At(i) <= 0) { err << "element of size can't be:" << size.At(i); }
  }

  if (pass_checked) {
    return Maybe<void>::Ok();
  } else {
    return oneflow::Error::CheckFailedError() << err.str();
  }
}

}  // namespace

REGISTER_USER_OP("affine_grid")
    .Input("theta")
    .Output("grid")
    .Attr<Shape>("size")
    .Attr<bool>("align_corners")
    .SetCheckAttrFn(CheckAttr)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& theta = ctx->InputTensorDesc("theta", 0);
      user_op::TensorDesc* grid = ctx->OutputTensorDesc("grid", 0);
      const Shape& size = ctx->Attr<Shape>("size");
      // Only support 2D or 3D affine grid with NCHW layout
      // For 2D grid: theta = { N, 2, 3 },
      //              size  = { N, C, H, W }
      //              grid  = { N, H, W, 2 }
      // For 3D grid: theta = { N, 3, 4 },
      //              size  = { N, C, D, H, W }
      //              grid  = { N, D, H, W, 3 }
      bool is_2d_grid = true;
      if (theta.shape().At(1) == 2) {
        CHECK_EQ_OR_RETURN(theta.shape().At(2), 3) << "Theta shape  MUST be (N, 2, 3) or (N, 3, 4)";
        CHECK_EQ_OR_RETURN(size.NumAxes(), 4) << "Dimension of size MUST be 4, when 2d affine grid";
        CHECK_EQ_OR_RETURN(theta.shape().At(0), size.At(0))
            << "Theta and size MUST have same batch dimension";
        is_2d_grid = true;
      } else if (theta.shape().At(1) == 3) {
        CHECK_EQ_OR_RETURN(theta.shape().At(2), 4) << "Theta shape  MUST be (N, 2, 3) or (N, 3, 4)";
        CHECK_EQ_OR_RETURN(size.NumAxes(), 5) "Dimension of size MUST be 4, when 3d affine grid";
        CHECK_EQ_OR_RETURN(theta.shape().At(0), size.At(0))
            << "Theta and size MUST have same batch dimension";
        is_2d_grid = false;
      } else {
        CHECK_OR_RETURN(false) << "Theta MUST be 2D or 3D grid";
      }
      *grid->mut_is_dynamic() = theta.is_dynamic();
      Shape& grid_shape = *grid->mut_shape();
      if (is_2d_grid) {
        grid_shape = {size.At(0), size.At(2), size.At(3), 2};
      } else {
        grid_shape = {size.At(0), size.At(2), size.At(3), size.At(4), 3};
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("theta", 0), 0)
          .Split(user_op::OpArg("grid", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("grid", 0) = ctx->InputDType("theta", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("affine_grid_grad")
    .Input("dgrid")
    .Output("dtheta")
    .Attr<Shape>("size")
    .Attr<bool>("align_corners")
    .SetCheckAttrFn(CheckAttr)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& size = ctx->Attr<Shape>("size");

      if (size.NumAxes() == 4) {
        *(ctx->OutputTensorDesc("dtheta", 0)->mut_shape()) = {size.At(0), 2, 3};
      } else if (size.NumAxes() == 5) {
        *(ctx->OutputTensorDesc("dtheta", 0)->mut_shape()) = {size.At(0), 3, 4};
      } else {
        CHECK_OR_RETURN(false) << "size MUST be 4D or 5D";
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dgrid", 0), 0)
          .Split(user_op::OpArg("dtheta", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dtheta", 0) = ctx->InputDType("dgrid", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("affine_grid")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("theta", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("affine_grid_grad")
                .Input("dgrid", op.GetGradTensorWithOpOutput("grid", 0))
                .Output("dtheta")
                .Attr("size", op.attr<Shape>("size"))
                .Attr("align_corners", op.attr<bool>("align_corners"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dtheta", 0), "theta", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
