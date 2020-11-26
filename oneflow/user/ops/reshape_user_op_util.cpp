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

namespace oneflow {

Maybe<Shape> ReshapeUserOpUtil::GetLogicalOutBlobShape(const Shape& in_shape,
                                                       const ShapeProto& reshape_proto) {
  size_t total_elem_dim_exclude_minus_1 = 1;
  bool has_minus_1 = false;
  bool minus_1_axis = -1;
  DimVector dim_vec;
  FOR_RANGE(int, axis, 0, reshape_proto.dim_size()) {
    int64_t dim = reshape_proto.dim(axis);
    dim_vec.push_back(dim);
    if (dim == -1) {
      CHECK_OR_RETURN(has_minus_1 == false) << "only one `-1' supported";
      has_minus_1 = true;
      minus_1_axis = axis;
    } else if (dim > 0) {
      CHECK_LE_OR_RETURN(dim, in_shape.elem_cnt()) << "invalid axis: " << axis << ", dim: " << dim;
      total_elem_dim_exclude_minus_1 *= dim;
      CHECK_LE_OR_RETURN(total_elem_dim_exclude_minus_1, in_shape.elem_cnt())
          << "element number in reshape_conf is bigger than input blob";
    } else {
      OF_UNIMPLEMENTED() << "only positive number or -1 supported";
    }
  }
  CHECK_EQ_OR_RETURN(in_shape.elem_cnt() % total_elem_dim_exclude_minus_1, 0);
  if (has_minus_1) {
    dim_vec[minus_1_axis] = in_shape.elem_cnt() / total_elem_dim_exclude_minus_1;
  } else {
    CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), total_elem_dim_exclude_minus_1)
        << "input blob's element number not equals reshape_conf";
  }
  return std::make_shared<Shape>(dim_vec);
}

Maybe<void> ReshapeUserOpUtil::Squeeze(const Shape& origin, Shape* shape,
                                       HashMap<int, int>* squeezed_axis2origin_axis) {
  CHECK_GT_OR_RETURN(origin.NumAxes(), 0);
  DimVector dim_vec;
  FOR_RANGE(int, axis, 0, origin.NumAxes()) {
    int64_t dim = origin.At(axis);
    CHECK_GT_OR_RETURN(dim, 0);
    if (dim == 1) { continue; }
    CHECK_OR_RETURN(squeezed_axis2origin_axis->emplace(dim_vec.size(), axis).second);
    dim_vec.push_back(dim);
  }
  *shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::GetGroupStartInAxis2OutAxis(
    const Shape& in_shape, const Shape& out_shape, const int64_t parallel_num,
    HashMap<int, int>* group_start_in_axis2out_axis) {
  CHECK_NE_OR_RETURN(in_shape.NumAxes(), 0);
  CHECK_NE_OR_RETURN(out_shape.NumAxes(), 0);
  CHECK_EQ(in_shape.elem_cnt(), out_shape.elem_cnt());
  int in_axis = in_shape.NumAxes() - 1;
  int out_axis = out_shape.NumAxes() - 1;
  while (in_axis >= 0 && out_axis >= 0) {
    if (in_shape.Count(in_axis) < out_shape.Count(out_axis)) {
      --in_axis;
    } else if (in_shape.Count(in_axis) > out_shape.Count(out_axis)) {
      --out_axis;
    } else {
      if (in_shape.At(in_axis) == out_shape.At(out_axis)
          || (in_shape.Count(in_axis) % parallel_num == 0
              && out_shape.Count(out_axis) % parallel_num == 0)) {
        (*group_start_in_axis2out_axis)[in_axis] = out_axis;
      }
      --in_axis;
      --out_axis;
    }
  }
  CHECK_GE_OR_RETURN(in_axis, -1);
  CHECK_GE_OR_RETURN(out_axis, -1);
  CHECK_LE_OR_RETURN(in_axis, 0);
  CHECK_LE_OR_RETURN(out_axis, 0);
  CHECK_EQ_OR_RETURN(in_axis == 0 && out_axis == 0, false);
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::GetReshapeUserOpSbpSignatures(const Shape& in_shape,
                                                             const Shape& out_shape,
                                                             std::vector<user_op::OpArg> in_args,
                                                             std::vector<user_op::OpArg> out_args,
                                                             user_op::SbpContext* ctx) {
  HashMap<int, int> squeezed_group_start_in_axis2out_axis;
  HashMap<int, int> in_squeezed_axis2original_axis;
  HashMap<int, int> out_squeezed_axis2original_axis;
  {
    Shape squeezed_in_shape;
    Shape squeezed_out_shape;
    ReshapeUserOpUtil::Squeeze(in_shape, &squeezed_in_shape, &in_squeezed_axis2original_axis);
    ReshapeUserOpUtil::Squeeze(out_shape, &squeezed_out_shape, &out_squeezed_axis2original_axis);
    ReshapeUserOpUtil::GetGroupStartInAxis2OutAxis(squeezed_in_shape, squeezed_out_shape,
                                                   ctx->parallel_num(),
                                                   &squeezed_group_start_in_axis2out_axis);
  }
  for (const auto& pair : squeezed_group_start_in_axis2out_axis) {
    int64_t start_in_axis = in_squeezed_axis2original_axis.at(pair.first);
    int64_t start_out_axis = out_squeezed_axis2original_axis.at(pair.second);
    ctx->NewBuilder().Split(in_args, start_in_axis).Split(out_args, start_out_axis).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
