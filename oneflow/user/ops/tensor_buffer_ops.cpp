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

REGISTER_CPU_ONLY_USER_OP("tensor_buffer_to_tensor")
    .Input("in")
    .Output("out")
    .Attr<Shape>("instance_shape")
    .Attr<DataType>("dtype")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      out->set_is_dynamic(in->is_dynamic());
      const auto& instance_shape = ctx->Attr<Shape>("instance_shape");
      DimVector dim_vec;
      dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin(), in->shape().dim_vec().cend());
      dim_vec.insert(dim_vec.end(), instance_shape.dim_vec().cbegin(),
                     instance_shape.dim_vec().cend());
      *out->mut_shape() = Shape(dim_vec);
      const auto data_type = ctx->Attr<DataType>("dtype");
      CHECK_OR_RETURN(IsPODDataType(data_type));
      *out->mut_data_type() = data_type;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("tensor_to_tensor_buffer")
    .Input("in")
    .Output("out")
    .Attr<int32_t>("instance_dims")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      CHECK_OR_RETURN(IsPODDataType(in->data_type()));
      const Shape& in_shape = in->shape();
      const auto& instance_dims = ctx->Attr<int32_t>("instance_dims");
      CHECK_LT_OR_RETURN(instance_dims, in_shape.NumAxes());
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      out->set_is_dynamic(in->is_dynamic());
      DimVector out_dim_vec;
      out_dim_vec.insert(out_dim_vec.end(), in_shape.dim_vec().cbegin(),
                         in_shape.dim_vec().cend() - instance_dims);
      *out->mut_shape() = Shape(out_dim_vec);
      *out->mut_data_type() = DataType::kTensorBuffer;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      const auto& instance_dims = ctx->Attr<int32_t>("instance_dims");
      CHECK_LE_OR_RETURN(instance_dims, in.shape().NumAxes());
      FOR_RANGE(int64_t, i, 0, in.shape().NumAxes() - instance_dims) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("gen_tensor_buffer")
    .Output("out")
    .Attr<Shape>("shape")
    .Attr<std::vector<Shape>>("shape_list")
    .Attr<std::vector<float>>("value_list")
    .Attr<DataType>("data_type")
    .Attr<bool>("dynamic_out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      const Shape& shape = ctx->Attr<Shape>("shape");
      const int64_t num_tensor_buffers = shape.elem_cnt();
      const std::vector<Shape>& shape_list = ctx->Attr<std::vector<Shape>>("shape_list");
      const std::vector<float>& value_list = ctx->Attr<std::vector<float>>("value_list");
      CHECK_EQ_OR_RETURN(num_tensor_buffers, shape_list.size());
      CHECK_EQ_OR_RETURN(num_tensor_buffers, value_list.size());
      *out->mut_shape() = shape;
      *out->mut_data_type() = DataType::kTensorBuffer;
      out->set_is_dynamic(ctx->Attr<bool>("dynamic_out"));
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("tensor_buffer_to_list_of_tensors")
    .Input("in")
    .OutputWithMinimum("out", 1)
    .Attr<Shape>("out_shape")
    .Attr<DataType>("out_dtype")
    .Attr<bool>("dynamic_out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      CHECK_GT_OR_RETURN(in->shape().elem_cnt(), 0);
      CHECK_EQ_OR_RETURN(in->data_type(), DataType::kTensorBuffer);
      CHECK_OR_RETURN(!in->is_dynamic());
      const Shape& out_shape = ctx->Attr<Shape>("out_shape");
      const DataType out_dtype = ctx->Attr<DataType>("out_dtype");
      CHECK_OR_RETURN(IsPODDataType(out_dtype));
      const bool dynamic_out = ctx->Attr<bool>("dynamic_out");
      int64_t num_tensor_buffers = in->shape().elem_cnt();
      for (int64_t i = 0; i < num_tensor_buffers; ++i) {
        user_op::TensorDesc* out_i = ctx->TensorDesc4ArgNameAndIndex("out", i);
        *out_i->mut_shape() = out_shape;
        *out_i->mut_data_type() = out_dtype;
        out_i->set_is_dynamic(dynamic_out);
      }
      return Maybe<void>::Ok();
    })
    .SetOutputArgModifyFn([](user_op::GetOutputArgModifier GetOutputArgModifierFn,
                             const user_op::UserOpConfWrapper& conf) {
      if (conf.attr<bool>("dynamic_out")) {
        FOR_RANGE(int64_t, i, 0, conf.output_size("out")) {
          user_op::OutputArgModifier* out_i_modifier = GetOutputArgModifierFn("out", i);
          CHECK(out_i_modifier != nullptr);
          out_i_modifier->set_header_infered_before_compute(false);
        }
      }
    });

}  // namespace

}  // namespace oneflow
