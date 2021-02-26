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

template<typename T>
bool PowerOfTwo(T x) {
  static_assert(std::is_integral<T>::value, "T must be integral");
  return x != 0 && (x & (x - 1)) == 0;
}

}  // namespace

REGISTER_CPU_ONLY_USER_OP("image_batch_align")
    .Input("in")
    .Output("out")
    .Attr<Shape>("shape")
    .Attr<DataType>("data_type")
    .Attr<int32_t>("alignment")
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& def,
                       const user_op::UserOpConfWrapper& conf) -> Maybe<void> {
      bool check_failed = false;
      std::stringstream err;
      err << "Illegal attr value for " << conf.op_type_name() << " op, op_name: " << conf.op_name();
      const Shape& shape = conf.attr<Shape>("shape");
      if (shape.NumAxes() != 3) {
        err << ", shape: " << shape.ToString() << " (image shape must has 3 axes)";
        check_failed = true;
      }
      DataType data_type = conf.attr<DataType>("data_type");
      if (data_type != DataType::kUInt8 && data_type != DataType::kFloat) {
        err << ", data_type: " << data_type << " (only support kUInt8 and kFloat for now)";
        check_failed = true;
      }
      int32_t alignment = conf.attr<int32_t>("alignment");
      if (alignment < 0) {
        err << ", alignment: " << alignment << " (alignment must be greater than or equal to 0)";
        check_failed = true;
      } else if (alignment != 0 && !PowerOfTwo(alignment)) {
        err << ", alignment: " << alignment
            << " (alignment must be power of 2 when it's not equal to 0)";
        check_failed = true;
      }
      if (check_failed) { return oneflow::Error::CheckFailedError() << err.str(); }
      return Maybe<void>::Ok();
    })
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      CHECK_OR_RETURN(in_desc->data_type() == DataType::kTensorBuffer);
      CHECK_OR_RETURN(in_desc->shape().NumAxes() == 1);
      const Shape& shape_attr = ctx->Attr<Shape>("shape");
      DimVector dim_vec(shape_attr.NumAxes() + 1);
      dim_vec.at(0) = in_desc->shape().elem_cnt();
      FOR_RANGE(int64_t, i, 0, shape_attr.NumAxes()) { dim_vec.at(i + 1) = shape_attr.At(i); }
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc->mut_shape() = Shape(dim_vec);
      *out_desc->mut_data_type() = ctx->Attr<DataType>("data_type");
      out_desc->set_is_dynamic(true);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetOutputArgModifyFn([](user_op::GetOutputArgModifier GetOutputArgModifierFn,
                             const user_op::UserOpConfWrapper& conf) {
      user_op::OutputArgModifier* out_modifier = GetOutputArgModifierFn("out", 0);
      CHECK(out_modifier != nullptr);
      out_modifier->set_header_infered_before_compute(false);
    });

}  // namespace oneflow
