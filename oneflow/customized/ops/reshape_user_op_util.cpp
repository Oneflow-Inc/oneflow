#include "oneflow/customized/ops/reshape_user_op_util.h"

namespace oneflow {
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
                                                             user_op::SbpContext* ctx) {
  HashMap<int, int> squeezed_group_start_in_axis2out_axis;
  HashMap<int, int> in_squeezed_axis2original_axis;
  HashMap<int, int> out_squeezed_axis2original_axis;
  {
    Shape squeezed_in_shape;
    Shape squeezed_out_shape;
    ReshapeUserOpUtil::Squeeze(in_shape, &squeezed_in_shape, &in_squeezed_axis2original_axis);
    ReshapeUserOpUtil::Squeeze(out_shape, &squeezed_out_shape, &out_squeezed_axis2original_axis);
    ReshapeUserOpUtil::GetGroupStartInAxis2OutAxis(squeezed_in_shape, squeezed_out_shape, ctx->parallel_num(),
                                               &squeezed_group_start_in_axis2out_axis);
  }
  for (const auto& pair : squeezed_group_start_in_axis2out_axis) {
    int64_t start_in_axis = in_squeezed_axis2original_axis.at(pair.first);
    int64_t start_out_axis = out_squeezed_axis2original_axis.at(pair.second);
    ctx->NewBuilder()
        .Split(ctx->inputs(), start_in_axis)
        .Split(ctx->outputs(), start_out_axis)
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(ctx->inputs())
      .PartialSum(ctx->outputs())
      .Build();
  return Maybe<void>::Ok();
}
}  // namespace oneflow
