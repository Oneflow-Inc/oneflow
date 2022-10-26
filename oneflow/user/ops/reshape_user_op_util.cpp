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
#include "oneflow/user/ops/reshape_user_op_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/cpp_attribute.h"

namespace oneflow {

Maybe<Shape> ReshapeUserOpUtil::GetLogicalOutBlobShape(const Shape& in_shape,
                                                       const Shape& reshape) {
  if (unlikely(in_shape.elem_cnt() == 0)) {
    FOR_RANGE(int, axis, 0, reshape.NumAxes()) {
      int64_t dim = reshape.At(axis);
      if (dim == -1) {
        return Error::RuntimeError()
               << "Cannot reshape tensor of 0 elements into shape " << reshape.DebugStr()
               << " because the unspecified dimension size -1 can be any value and is ambiguous";
      } else if (dim < 0) {
        return Error::RuntimeError() << "Invalid shape dimension " << dim
                                     << ", the shape dimension can not to be less than 0";
      }
    }
    return std::make_shared<Shape>(reshape);
  }
  size_t total_elem_dim_exclude_minus_1 = 1;
  bool has_minus_1 = false;
  bool minus_1_axis = -1;
  DimVector dim_vec;
  FOR_RANGE(int, axis, 0, reshape.NumAxes()) {
    int64_t dim = reshape.At(axis);
    dim_vec.emplace_back(dim);
    if (dim == -1) {
      CHECK_OR_RETURN(has_minus_1 == false)
          << Error::RuntimeError()
          << "There are multiple '-1' in the shape list, only one '-1' can be inferred";
      has_minus_1 = true;
      minus_1_axis = axis;
    } else if (dim > 0) {
      CHECK_LE_OR_RETURN(dim, in_shape.elem_cnt())
          << Error::RuntimeError() << "Invalid axis: " << axis << ", dim: " << dim;
      total_elem_dim_exclude_minus_1 *= dim;
      CHECK_LE_OR_RETURN(total_elem_dim_exclude_minus_1, in_shape.elem_cnt())
          << Error::RuntimeError()
          << "Element number in reshape_conf must be less than or equal to input blob, "
          << "but got " << total_elem_dim_exclude_minus_1 << " and " << in_shape.elem_cnt();
    } else {
      OF_UNIMPLEMENTED() << "only positive number or -1 supported";
    }
  }
  CHECK_EQ_OR_RETURN(in_shape.elem_cnt() % total_elem_dim_exclude_minus_1, 0)
      << Error::RuntimeError()
      << "Element number in input blob must be an integer multiple of reshape_conf, "
      << "but got " << in_shape.elem_cnt() << " and " << total_elem_dim_exclude_minus_1;
  if (has_minus_1) {
    dim_vec[minus_1_axis] = in_shape.elem_cnt() / total_elem_dim_exclude_minus_1;
  } else {
    CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), total_elem_dim_exclude_minus_1)
        << "Element number in input blob must be equal to reshape_conf, "
        << "but got " << in_shape.elem_cnt() << " and " << total_elem_dim_exclude_minus_1;
  }
  return std::make_shared<Shape>(dim_vec);
}

Maybe<void> ReshapeUserOpUtil::Squeeze(const Shape& origin, Shape* shape,
                                       HashMap<int, int>* squeezed_axis2origin_axis) {
  DimVector dim_vec;
  FOR_RANGE(int, axis, 0, origin.NumAxes()) {
    int64_t dim = origin.At(axis);
    CHECK_GE_OR_RETURN(dim, 0) << Error::RuntimeError()
                               << "Trying to suqeeze tensor with negative dimension " << dim
                               << " : " << origin.DebugStr();
    if (dim == 1) { continue; }
    CHECK_OR_RETURN(squeezed_axis2origin_axis->emplace(dim_vec.size(), axis).second)
        << "emplace error";  // NOLINT(maybe-need-error-msg)
    dim_vec.emplace_back(dim);
  }
  *shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::GetGroupStartInAxis2OutAxis(
    const Shape& in_shape, const Shape& out_shape, const int64_t hierarchy_value,
    HashMap<int, int>* group_start_in_axis2out_axis) {
  CHECK_GE_OR_RETURN(in_shape.NumAxes(), 0)
      << Error::RuntimeError()
      << "The dimension of input tensor must be greater than or equal to zero, "
      << "but got " << in_shape.NumAxes();  // support 0D tensor
  CHECK_GE_OR_RETURN(out_shape.NumAxes(), 0)
      << Error::RuntimeError()
      << "The dimension of output tensor must be greater than or equal to zero, "
      << "but got " << out_shape.NumAxes();  // support 0D tensor
  CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), out_shape.elem_cnt())
      << Error::RuntimeError()
      << "The element number of input tensor must be equal to output tensor, "
      << "but got " << in_shape.elem_cnt() << " and " << out_shape.elem_cnt();
  // Initialization
  // shape_count is the product of the axis length in [start_axis, end)
  int64_t in_shape_count = 1;
  int64_t out_shape_count = 1;
  int64_t in_axis = in_shape.NumAxes();
  int64_t out_axis = out_shape.NumAxes();
  // Move forward functions
  auto Move2NextAxis = [](const Shape& shape, int64_t* axis, int64_t* shape_count) {
    (*axis)--;
    if (*axis >= 0) { *shape_count *= shape.At(*axis); }
  };
  auto MoveInAxis = [&] { Move2NextAxis(in_shape, &in_axis, &in_shape_count); };
  auto MoveOutAxis = [&] { Move2NextAxis(out_shape, &out_axis, &out_shape_count); };
  // Move the first step
  MoveInAxis();
  MoveOutAxis();
  // At the last step, both in_axis == out_axis == 0
  // Then they would move to -1 simultaneously.
  while (in_axis >= 0) {
    if (in_shape_count == out_shape_count) {
      // Record split axises
      if (in_shape.At(in_axis) == out_shape.At(out_axis)
          || (in_shape.At(in_axis) % hierarchy_value == 0
              && out_shape.At(out_axis) % hierarchy_value == 0)) {
        (*group_start_in_axis2out_axis)[in_axis] = out_axis;
      }
      // Move forward
      MoveInAxis();
      MoveOutAxis();
    } else if (in_shape_count < out_shape_count) {
      MoveInAxis();
    } else {
      // in_shape_count > out_shape_count
      MoveOutAxis();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::GetReshapeUserOpSbpSignatures(
    const Shape& in_shape, const Shape& out_shape, std::vector<user_op::OpArg> in_args,
    std::vector<user_op::OpArg> out_args, const int64_t hierarchy_value,
    user_op::UserOpSbpSignatureBuilder* builder) {
  if (in_shape.NumAxes() == 0 || in_shape.elem_cnt() == 0) {
    return Maybe<void>::Ok();
  }  // 0D/0Size tensor only support b2b
  HashMap<int, int> squeezed_group_start_in_axis2out_axis;
  HashMap<int, int> in_squeezed_axis2original_axis;
  HashMap<int, int> out_squeezed_axis2original_axis;
  {
    Shape squeezed_in_shape;
    Shape squeezed_out_shape;
    JUST(ReshapeUserOpUtil::Squeeze(in_shape, &squeezed_in_shape, &in_squeezed_axis2original_axis));
    JUST(ReshapeUserOpUtil::Squeeze(out_shape, &squeezed_out_shape,
                                    &out_squeezed_axis2original_axis));
    JUST(ReshapeUserOpUtil::GetGroupStartInAxis2OutAxis(squeezed_in_shape, squeezed_out_shape,
                                                        hierarchy_value,
                                                        &squeezed_group_start_in_axis2out_axis));
  }
  for (const auto& pair : squeezed_group_start_in_axis2out_axis) {
    int64_t start_in_axis = in_squeezed_axis2original_axis.at(pair.first);
    int64_t start_out_axis = out_squeezed_axis2original_axis.at(pair.second);
    builder->Split(in_args, start_in_axis).Split(out_args, start_out_axis).Build();
  }
  builder->PartialSum(in_args).PartialSum(out_args).Build();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
