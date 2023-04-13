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
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

Maybe<void> CheckAttr_(const user_op::UserOpDefWrapper& def,
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

/* static */ Maybe<void> AffineGridOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& theta = ctx->InputTensorDesc("theta", 0);
  user_op::TensorDesc* grid = ctx->MutOutputTensorDesc("grid", 0);
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
    CHECK_EQ_OR_RETURN(size.NumAxes(), 5) << "Dimension of size MUST be 4, when 3d affine grid";
    CHECK_EQ_OR_RETURN(theta.shape().At(0), size.At(0))
        << "Theta and size MUST have same batch dimension";
    is_2d_grid = false;
  } else {
    CHECK_OR_RETURN(false) << "Theta MUST be 2D or 3D grid";
  }
  grid->set_is_dynamic(theta.is_dynamic());
  Shape grid_shape;
  if (is_2d_grid) {
    grid_shape = {size.At(0), size.At(2), size.At(3), 2};
  } else {
    grid_shape = {size.At(0), size.At(2), size.At(3), size.At(4), 3};
  }
  grid->set_shape(grid_shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AffineGridOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& theta = ctx->InputTensorDesc("theta", 0);
  user_op::TensorDesc* grid = ctx->MutOutputTensorDesc("grid", 0);
  const Shape& size = ctx->Attr<Shape>("size");
  // Only support 2D or 3D affine grid with NCHW layout
  // For 2D grid: theta = { N, 2, 3 },
  //              size  = { N, C, H, W }
  //              grid  = { N, H, W, 2 }
  // For 3D grid: theta = { N, 3, 4 },
  //              size  = { N, C, D, H, W }
  //              grid  = { N, D, H, W, 3 }
  const Shape& theta_shape = theta.shape();
  bool is_2d_grid = true;
  if (theta_shape.At(1) == 2) {
    CHECK_EQ_OR_RETURN(theta_shape.At(2), 3) << "Theta shape  MUST be (N, 2, 3) or (N, 3, 4)";
    CHECK_EQ_OR_RETURN(size.NumAxes(), 4) << "Dimension of size MUST be 4, when 2d affine grid";
    is_2d_grid = true;
  } else if (theta_shape.At(1) == 3) {
    CHECK_EQ_OR_RETURN(theta_shape.At(2), 4) << "Theta shape  MUST be (N, 2, 3) or (N, 3, 4)";
    CHECK_EQ_OR_RETURN(size.NumAxes(), 5) << "Dimension of size MUST be 4, when 3d affine grid";
    is_2d_grid = false;
  } else {
    CHECK_OR_RETURN(false) << "Theta MUST be 2D or 3D grid";
  }

  int64_t N = size.At(0);
  const int64_t& parallel_num = ctx->parallel_ctx().parallel_num();
  if (parallel_num > 1) {
    const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("theta", 0);
    Shape logical_shape = theta_shape;
    logical_shape.Set(0, size.At(0));
    const auto& physical_shape =
        JUST(GetPhysicalShape(logical_shape, nd_sbp, ctx->parallel_desc(), ctx->parallel_ctx()));
    N = physical_shape->At(0);
  }
  CHECK_EQ_OR_RETURN(theta_shape.At(0), N)
      << "The dimension 0 size of theta shape should be " << N << ", but got " << theta_shape.At(0);

  grid->set_is_dynamic(theta.is_dynamic());
  Shape grid_shape;
  if (is_2d_grid) {
    grid_shape = {N, size.At(2), size.At(3), 2};
  } else {
    grid_shape = {N, size.At(2), size.At(3), size.At(4), 3};
  }
  grid->set_shape(grid_shape);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AffineGridOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("theta", 0), 0)
      .Split(user_op::OpArg("grid", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AffineGridOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                 const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_(def, conf);
}

/* static */ Maybe<void> AffineGridOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("grid", 0, ctx->InputDType("theta", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AffineGridGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dgrid = ctx->InputTensorDesc("dgrid", 0);
  const Shape& size = ctx->Attr<Shape>("size");
  if (size.NumAxes() == 4) {
    ctx->MutOutputTensorDesc("dtheta", 0)->set_shape(Shape({dgrid.shape().At(0), 2, 3}));
  } else if (size.NumAxes() == 5) {
    ctx->MutOutputTensorDesc("dtheta", 0)->set_shape(Shape({dgrid.shape().At(0), 3, 4}));
  } else {
    CHECK_OR_RETURN(false) << "size MUST be 4D or 5D";
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> AffineGridGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> AffineGridGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dgrid", 0), 0)
      .Split(user_op::OpArg("dtheta", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> AffineGridGradOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                     const user_op::UserOpConfWrapper& conf) {
  return CheckAttr_(def, conf);
}

/* static */ Maybe<void> AffineGridGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dtheta", 0, ctx->InputDType("dgrid", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
