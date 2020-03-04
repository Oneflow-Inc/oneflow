#include "oneflow/core/operator/reshape_op_util.h"

namespace oneflow {
Maybe<Shape> ReshapeOpUtil::GetLogicalOutBlobShape(const Shape& in_shape,
                                                   const ShapeProto& reshape_proto) {
  size_t total_elem_dim_exclude_minus_1 = 1;
  bool has_minus_1 = false;
  bool minus_1_axis = -1;
  DimVector dim_vec;
  FOR_RANGE(int, axis, 0, reshape_proto.dim_size()) {
    int64_t dim = reshape_proto.dim(axis);
    dim_vec.push_back(dim);
    if (dim == -1) {
      OF_CHECK(has_minus_1 == false) << "only one `-1' supported";
      has_minus_1 = true;
      minus_1_axis = axis;
    } else if (dim > 0) {
      OF_CHECK_LE(dim, in_shape.elem_cnt()) << "invalid axis: " << axis << ", dim: " << dim;
      total_elem_dim_exclude_minus_1 *= dim;
      OF_CHECK_LE(total_elem_dim_exclude_minus_1, in_shape.elem_cnt())
          << "element number in reshape_conf is bigger than input blob";
    } else {
      OF_UNIMPLEMENTED() << "only positive number or -1 supported";
    }
  }
  OF_CHECK_EQ(in_shape.elem_cnt() % total_elem_dim_exclude_minus_1, 0);
  if (has_minus_1) {
    dim_vec[minus_1_axis] = in_shape.elem_cnt() / total_elem_dim_exclude_minus_1;
  } else {
    OF_CHECK_EQ(in_shape.elem_cnt(), total_elem_dim_exclude_minus_1)
        << "input blob's element number not equals reshape_conf";
  }
  return std::make_shared<Shape>(dim_vec);
}

Maybe<void> ReshapeOpUtil::Squeeze(const Shape& origin, Shape* shape,
                                   HashMap<int, int>* squeezed_axis2origin_axis) {
  OF_CHECK_GT(origin.NumAxes(), 0);
  DimVector dim_vec;
  FOR_RANGE(int, axis, 0, origin.NumAxes()) {
    int64_t dim = origin.At(axis);
    OF_CHECK_GT(dim, 0);
    if (dim == 1) { continue; }
    OF_CHECK(squeezed_axis2origin_axis->emplace(dim_vec.size(), axis).second);
    dim_vec.push_back(dim);
  }
  *shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeOpUtil::GetGroupStartInAxis2OutAxis(
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
  OF_CHECK_GE(in_axis, -1);
  OF_CHECK_GE(out_axis, -1);
  OF_CHECK_LE(in_axis, 0);
  OF_CHECK_LE(out_axis, 0);
  OF_CHECK_EQ(in_axis == 0 && out_axis == 0, false);
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeOpUtil::GetReshapeSbpSignatures(const Shape& in_shape, const Shape& out_shape,
                                                   const PbRpf<std::string>& input_bns,
                                                   const PbRpf<std::string>& output_bns,
                                                   const int64_t parallel_num,
                                                   SbpSignatureList* sbp_sig_list) {
  HashMap<int, int> squeezed_group_start_in_axis2out_axis;
  HashMap<int, int> in_squeezed_axis2original_axis;
  HashMap<int, int> out_squeezed_axis2original_axis;
  {
    Shape squeezed_in_shape;
    Shape squeezed_out_shape;
    ReshapeOpUtil::Squeeze(in_shape, &squeezed_in_shape, &in_squeezed_axis2original_axis);
    ReshapeOpUtil::Squeeze(out_shape, &squeezed_out_shape, &out_squeezed_axis2original_axis);
    ReshapeOpUtil::GetGroupStartInAxis2OutAxis(squeezed_in_shape, squeezed_out_shape, parallel_num,
                                               &squeezed_group_start_in_axis2out_axis);
  }
  for (const auto& pair : squeezed_group_start_in_axis2out_axis) {
    int64_t start_in_axis = in_squeezed_axis2original_axis.at(pair.first);
    int64_t start_out_axis = out_squeezed_axis2original_axis.at(pair.second);
    SbpSignatureBuilder()
        .Split(input_bns, start_in_axis)
        .Split(output_bns, start_out_axis)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .PartialSum(input_bns)
      .PartialSum(output_bns)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}
}  // namespace oneflow
