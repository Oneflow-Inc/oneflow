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

REGISTER_USER_OP("upsample_linear_1d")
    .Input("x")
    .Output("y")
    .Attr<float>("scale_factor")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
      const float scale_factor = ctx->Attr<float>("scale_factor");

      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && x_desc->shape().NumAxes() == 3)
          << "upsample_linear_1d only supports NCH";
      *y_desc->mut_shape() = Shape({x_desc->shape().At(0), x_desc->shape().At(1),
                                    static_cast<int32_t>(scale_factor * x_desc->shape().At(2))});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_nearest_1d")
    .Input("x")
    .Output("y")
    .Attr<float>("scale_factor")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
      const float scale_factor = ctx->Attr<float>("scale_factor");
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && x_desc->shape().NumAxes() == 3)
          << "upsample_nearest_1d only supports NCH";
      *y_desc->mut_shape() = Shape({x_desc->shape().At(0), x_desc->shape().At(1),
                                    static_cast<int32_t>(scale_factor * x_desc->shape().At(2))});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_nearest_2d")
    .Input("x")
    .Output("y")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
      const float height_scale = ctx->Attr<float>("height_scale");
      const float width_scale = ctx->Attr<float>("width_scale");
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && x_desc->shape().NumAxes() == 4)
          << "upsample_nearest_2d only supports NCHW";
      *y_desc->mut_shape() = Shape({x_desc->shape().At(0), x_desc->shape().At(1),
                                    static_cast<int32_t>(height_scale * x_desc->shape().At(2)),
                                    static_cast<int32_t>(width_scale * x_desc->shape().At(3))});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_bilinear_2d")
    .Input("x")
    .Output("y")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
      const float height_scale = ctx->Attr<float>("height_scale");
      const float width_scale = ctx->Attr<float>("width_scale");
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && x_desc->shape().NumAxes() == 4)
          << "upsample_bilinear_2d only supports NCHW";
      *y_desc->mut_shape() = Shape({x_desc->shape().At(0), x_desc->shape().At(1),
                                    static_cast<int32_t>(height_scale * x_desc->shape().At(2)),
                                    static_cast<int32_t>(width_scale * x_desc->shape().At(3))});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_bicubic_2d")
    .Input("x")
    .Output("y")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
      const float height_scale = ctx->Attr<float>("height_scale");
      const float width_scale = ctx->Attr<float>("width_scale");
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && x_desc->shape().NumAxes() == 4)
          << "upsample_bicubic_2d only supports NCHW";
      *y_desc->mut_shape() = Shape({x_desc->shape().At(0), x_desc->shape().At(1),
                                    static_cast<int32_t>(height_scale * x_desc->shape().At(2)),
                                    static_cast<int32_t>(width_scale * x_desc->shape().At(3))});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample")
    .Input("x")
    .Output("y")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .Attr<std::string>("interpolation")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
      user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
      const float height_scale = ctx->Attr<float>("height_scale");
      const float width_scale = ctx->Attr<float>("width_scale");
      if (ctx->Attr<std::string>("data_format") != "channels_first"
          || x_desc.shape().NumAxes() != 4) {
        LOG(FATAL) << "upsample only supports NCHW";
      }
      *y_desc->mut_shape() = Shape({x_desc.shape().At(0), x_desc.shape().At(1),
                                    static_cast<int32_t>(height_scale * x_desc.shape().At(2)),
                                    static_cast<int32_t>(width_scale * x_desc.shape().At(3))});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_nearest_3d")
    .Input("x")
    .Output("y")
    .Attr<float>("depth_scale")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
      const float depth_scale = ctx->Attr<float>("depth_scale");
      const float height_scale = ctx->Attr<float>("height_scale");
      const float width_scale = ctx->Attr<float>("width_scale");
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && x_desc->shape().NumAxes() == 5)
          << "upsample_nearest_3d only supports NCDHW";
      *y_desc->mut_shape() = Shape({x_desc->shape().At(0), x_desc->shape().At(1),
                                    static_cast<int32_t>(depth_scale * x_desc->shape().At(2)),
                                    static_cast<int32_t>(height_scale * x_desc->shape().At(3)),
                                    static_cast<int32_t>(width_scale * x_desc->shape().At(4))});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_trilinear_3d")
    .Input("x")
    .Output("y")
    .Attr<float>("depth_scale")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
      user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
      const float depth_scale = ctx->Attr<float>("depth_scale");
      const float height_scale = ctx->Attr<float>("height_scale");
      const float width_scale = ctx->Attr<float>("width_scale");
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && x_desc->shape().NumAxes() == 5)
          << "upsample_trilinear_3d only supports NCDHW";
      *y_desc->mut_shape() = Shape({x_desc->shape().At(0), x_desc->shape().At(1),
                                    static_cast<int32_t>(depth_scale * x_desc->shape().At(2)),
                                    static_cast<int32_t>(height_scale * x_desc->shape().At(3)),
                                    static_cast<int32_t>(width_scale * x_desc->shape().At(4))});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("y", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_linear_1d_grad")
    .Input("dy")
    .Input("x")
    .Output("dx")
    .Attr<float>("scale_factor")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& dy_shape = ctx->InputShape("dy", 0);
      Shape* dx_shape = ctx->OutputShape("dx", 0);
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && dy_shape.NumAxes() == 3)
          << "upsample_linear_1d_grad only supports NCH";
      *dx_shape = ctx->InputShape("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_nearest_1d_grad")
    .Input("dy")
    .Input("x")
    .Output("dx")
    .Attr<float>("scale_factor")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& dy_shape = ctx->InputShape("dy", 0);
      Shape* dx_shape = ctx->OutputShape("dx", 0);
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && dy_shape.NumAxes() == 3)
          << "upsample_nearest_1d_grad only supports NCH";
      *dx_shape = ctx->InputShape("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_nearest_2d_grad")
    .Input("dy")
    .Input("x")
    .Output("dx")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& dy_shape = ctx->InputShape("dy", 0);
      Shape* dx_shape = ctx->OutputShape("dx", 0);
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && dy_shape.NumAxes() == 4)
          << "upsample_nearest_2d_grad only supports NCHW";
      *dx_shape = ctx->InputShape("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_bilinear_2d_grad")
    .Input("dy")
    .Input("x")
    .Output("dx")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& dy_shape = ctx->InputShape("dy", 0);
      Shape* dx_shape = ctx->OutputShape("dx", 0);
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && dy_shape.NumAxes() == 4)
          << "upsample_bilinear_2d_grad only supports NCHW";
      *dx_shape = ctx->InputShape("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_bicubic_2d_grad")
    .Input("dy")
    .Input("x")
    .Output("dx")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& dy_shape = ctx->InputShape("dy", 0);
      Shape* dx_shape = ctx->OutputShape("dx", 0);
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && dy_shape.NumAxes() == 4)
          << "upsample_bicubic_2d_grad only supports NCHW";
      *dx_shape = ctx->InputShape("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_grad")
    .Input("dy")
    .Input("x")
    .Output("dx")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .Attr<std::string>("interpolation")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& dy_shape = ctx->InputShape("dy", 0);
      Shape* dx_shape = ctx->OutputShape("dx", 0);
      if (ctx->Attr<std::string>("data_format") != "channels_first" || dy_shape.NumAxes() != 4) {
        LOG(FATAL) << "upsample_nearest only supports NCHW";
      }
      *dx_shape = ctx->InputShape("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_nearest_3d_grad")
    .Input("dy")
    .Input("x")
    .Output("dx")
    .Attr<float>("depth_scale")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& dy_shape = ctx->InputShape("dy", 0);
      Shape* dx_shape = ctx->OutputShape("dx", 0);
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && dy_shape.NumAxes() == 5)
          << "upsample_nearest_3d_grad only supports NCDHW";
      *dx_shape = ctx->InputShape("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("upsample_trilinear_3d_grad")
    .Input("dy")
    .Input("x")
    .Output("dx")
    .Attr<float>("depth_scale")
    .Attr<float>("height_scale")
    .Attr<float>("width_scale")
    .Attr<bool>("align_corners")
    .Attr<std::string>("data_format")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& dy_shape = ctx->InputShape("dy", 0);
      Shape* dx_shape = ctx->OutputShape("dx", 0);
      CHECK_OR_RETURN(ctx->Attr<std::string>("data_format") == "channels_first"
                      && dy_shape.NumAxes() == 5)
          << "upsample_trilinear_3d_grad only supports NCDHW";
      *dx_shape = ctx->InputShape("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("dy", 0), 0).Split(user_op::OpArg("dx", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("upsample_linear_1d")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("upsample_linear_1d_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Attr("scale_factor", op.attr<float>("scale_factor"))
                .Attr("align_corners", op.attr<bool>("align_corners"))
                .Attr("data_format", op.attr<std::string>("data_format"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("upsample_nearest_1d")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("upsample_nearest_1d_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Attr("scale_factor", op.attr<float>("scale_factor"))
                .Attr("data_format", op.attr<std::string>("data_format"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("upsample_nearest_2d")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("upsample_nearest_2d_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Attr("height_scale", op.attr<float>("height_scale"))
                .Attr("width_scale", op.attr<float>("width_scale"))
                .Attr("data_format", op.attr<std::string>("data_format"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("upsample_bilinear_2d")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("upsample_bilinear_2d_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Attr("height_scale", op.attr<float>("height_scale"))
                .Attr("width_scale", op.attr<float>("width_scale"))
                .Attr("align_corners", op.attr<bool>("align_corners"))
                .Attr("data_format", op.attr<std::string>("data_format"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("upsample_bicubic_2d")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("upsample_bicubic_2d_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Attr("height_scale", op.attr<float>("height_scale"))
                .Attr("width_scale", op.attr<float>("width_scale"))
                .Attr("align_corners", op.attr<bool>("align_corners"))
                .Attr("data_format", op.attr<std::string>("data_format"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("upsample")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("upsample_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Attr("height_scale", op.attr<float>("height_scale"))
                .Attr("width_scale", op.attr<float>("width_scale"))
                .Attr("align_corners", op.attr<bool>("align_corners"))
                .Attr("data_format", op.attr<std::string>("data_format"))
                .Attr("interpolation", op.attr<std::string>("interpolation"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("upsample_nearest_3d")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("upsample_nearest_3d_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Attr("depth_scale", op.attr<float>("depth_scale"))
                .Attr("height_scale", op.attr<float>("height_scale"))
                .Attr("width_scale", op.attr<float>("width_scale"))
                .Attr("data_format", op.attr<std::string>("data_format"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("upsample_trilinear_3d")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("upsample_trilinear_3d_grad")
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Attr("depth_scale", op.attr<float>("depth_scale"))
                .Attr("height_scale", op.attr<float>("height_scale"))
                .Attr("width_scale", op.attr<float>("width_scale"))
                .Attr("align_corners", op.attr<bool>("align_corners"))
                .Attr("data_format", op.attr<std::string>("data_format"))
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
