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
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

void PrintSbpLog(SbpSignatureList* sbp_list) {
  for (const auto& sbp_sign : sbp_list->sbp_signature()) {
    std::cout << "cclog: one sbp sign: ";
    for (const auto& pair : sbp_sign.bn_in_op2sbp_parallel()) {
      std::cout << " bn: " << pair.first;
      if (pair.second.has_split_parallel()) {
        std::cout << " Split axis = " << pair.second.split_parallel().axis();
      } else if (pair.second.has_broadcast_parallel()) {
        std::cout << " Broadcast ";
      } else if (pair.second.has_partial_sum_parallel()) {
        std::cout << " PartialSum ";
      } else {
        std::cout << " ERROR !";
      }
    }
    std::cout << std::endl;
  }
}

}  // namespace

REGISTER_USER_OP("ccrelu")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("ccrelu_grad")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      CHECK(*dy_shape == *y_shape);
      *dx_shape = *y_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("y", 0), 0)
          .Split(user_op::OpArg("dy", 0), 0)
          .Split(user_op::OpArg("dx", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("ccrelu").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper ccrelu_grad_op =
        builder.Op("ccrelu_grad")
            .Input("y", op.output("out", 0))
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
            .Output("dx")
            .Build();
    op.BindGradTensorWithOpInput(ccrelu_grad_op.output("dx", 0), "in", 0);
    AddOp(ccrelu_grad_op);
  }
});

REGISTER_USER_OP("TestReshape")
    .Input("in")
    .Output("out")
    .Attr<Shape>("shape")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      const Shape& conf_shape = ctx->Attr<Shape>("shape");
      CHECK_EQ(in_shape->NumAxes(), conf_shape.NumAxes());
      *out_shape = conf_shape;
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("TestSource")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = Shape({5});
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kFloat;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("TestMultiOutputOrder")
    .Input("in")
    .Output("out1")
    .Output("out2")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out1_shape = ctx->Shape4ArgNameAndIndex("out1", 0);
      Shape* out2_shape = ctx->Shape4ArgNameAndIndex("out2", 0);
      *out1_shape = *in_shape;
      *out2_shape = *in_shape;
      int32_t last_axis = in_shape->NumAxes() - 1;
      out2_shape->Set(last_axis, in_shape->At(last_axis) * 2);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("TestSourceMultiGpuFixedOutNum")
    .Output("out")
    .Attr<int64_t>("out_num")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      int64_t out_num = ctx->Attr<int64_t>("out_num");
      const ParallelContext& parallel_ctx = ctx->parallel_ctx();
      BalancedSplitter bs(out_num, parallel_ctx.parallel_num());
      *out_shape = Shape({bs.At(parallel_ctx.parallel_id()).size()});

      const SbpParallel& out_sbp = ctx->SbpParallel4ArgNameAndIndex("out", 0);
      CHECK(out_sbp.has_split_parallel() && out_sbp.split_parallel().axis() == 0);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kFloat;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      int64_t parallel_num = ctx->parallel_num();
      DeviceType device_type = ctx->device_type();
      if (device_type == DeviceType::kCPU && parallel_num > 1) {
        ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("TestMultiInput")
    .Input("x1")
    .Input("x2")
    .Output("y")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x1_shape = ctx->Shape4ArgNameAndIndex("x1", 0);
      Shape* x2_shape = ctx->Shape4ArgNameAndIndex("x2", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      CHECK(*x1_shape == *x2_shape);
      *y_shape = *x1_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x1_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x1", 0);
      FOR_RANGE(int64_t, i, 0, x1_tensor.shape().NumAxes()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("TestMultiInputGrad")
    .Input("x1")
    .Input("x2")
    .Input("y_diff")
    .Output("x1_diff")
    .Output("x2_diff")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x1_shape = ctx->Shape4ArgNameAndIndex("x1", 0);
      Shape* x2_shape = ctx->Shape4ArgNameAndIndex("x2", 0);
      Shape* x1_diff_shape = ctx->Shape4ArgNameAndIndex("x1_diff", 0);
      Shape* x2_diff_shape = ctx->Shape4ArgNameAndIndex("x2_diff", 0);
      *x1_diff_shape = *x1_shape;
      *x2_diff_shape = *x2_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x1_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x1", 0);
      FOR_RANGE(int64_t, i, 0, x1_tensor.shape().NumAxes()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("TestMultiInput")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x1", 0) || op.NeedGenGradTensor4OpInput("x2", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper test_multi_input_grad_op =
            builder.Op("TestMultiInputGrad")
                .Input("x1", op.input("x1", 0))
                .Input("x2", op.input("x2", 0))
                .Input("y_diff", op.GetGradTensorWithOpOutput("y", 0))
                .Output("x1_diff")
                .Output("x2_diff")
                .Build();
        op.BindGradTensorWithOpInput(test_multi_input_grad_op.output("x1_diff", 0), "x1", 0);
        op.BindGradTensorWithOpInput(test_multi_input_grad_op.output("x2_diff", 0), "x2", 0);
        AddOp(test_multi_input_grad_op);
      }
    });

REGISTER_USER_OP("TestDynamicSource")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_tensor->mut_shape() = Shape({5});
      *out_tensor->mut_data_type() = DataType::kFloat;
      out_tensor->set_is_dynamic(true);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetOutputArgModifyFn([](user_op::GetOutputArgModifier GetOutputArgModifierFn,
                             const user_op::UserOpConfWrapper& conf) {
      user_op::OutputArgModifier* out_modifier = GetOutputArgModifierFn("out", 0);
      CHECK(out_modifier != nullptr);
      out_modifier->set_header_infered_before_compute(false);
    });

REGISTER_USER_OP("TestRandomSource")
    .Output("out")
    .Attr<int64_t>("seed")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_tensor->mut_shape() = Shape({5});
      *out_tensor->mut_data_type() = DataType::kFloat;
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("TestDataTypeAttr")
    .Input("in")
    .Output("out")
    .Attr<DataType>("output_type")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      *ctx->Dtype4ArgNameAndIndex("out", 0) = ctx->Attr<DataType>("output_type");
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("TestListDataTypeAndListShapeAndListStringAttr")
    .Input("in")
    .Output("out", 3)
    .Attr<std::vector<Shape>>("out_shapes")
    .Attr<std::vector<DataType>>("out_types")
    .Attr<std::vector<std::string>>("string_list")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const auto& out_shapes = ctx->Attr<std::vector<Shape>>("out_shapes");
      const auto& out_types = ctx->Attr<std::vector<DataType>>("out_types");
      const auto& string_list = ctx->Attr<std::vector<std::string>>("string_list");
      FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
        *ctx->Shape4ArgNameAndIndex("out", i) = out_shapes.at(i);
        *ctx->Dtype4ArgNameAndIndex("out", i) = out_types.at(i);
      }
      CHECK_GT_OR_RETURN(string_list.size(), 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("test_user_op_attr_auto_type")
    .Input("in")
    .Output("out")
    .Attr<int32_t>("int1")
    .Attr<int32_t>("int2")
    .SetTensorDescInferFn(user_op::TensorDescInferFnUtil::Unchanged);

REGISTER_CPU_ONLY_USER_OP("cpu_only_relu_test")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const auto* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      auto* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *in_desc;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
