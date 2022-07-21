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
    const Shape& in_shape, const Shape& out_shape, const int64_t parallel_num,
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
  CHECK_GE_OR_RETURN(in_axis, -1);                           // NOLINT(maybe-need-error-msg)
  CHECK_GE_OR_RETURN(out_axis, -1);                          // NOLINT(maybe-need-error-msg)
  CHECK_LE_OR_RETURN(in_axis, 0);                            // NOLINT(maybe-need-error-msg)
  CHECK_LE_OR_RETURN(out_axis, 0);                           // NOLINT(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(in_axis == 0 && out_axis == 0, false);  // NOLINT(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::GetReshapeUserOpSbpSignatures(
    const Shape& in_shape, const Shape& out_shape, std::vector<user_op::OpArg> in_args,
    std::vector<user_op::OpArg> out_args, const int64_t parallel_num,
    user_op::UserOpSbpSignatureBuilder* builder) {
  if (in_shape.NumAxes() == 0) { return Maybe<void>::Ok(); }
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
                                                        parallel_num,
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

namespace {

Maybe<void> GetInputNdSbp(user_op::InferNdSbpFnContext* ctx, const user_op::OpArg& in_arg,
                          NdSbp* distribution) {
  *distribution = ctx->NdSbpHint4InputArgNameAndIndex(in_arg.name(), in_arg.index());
  const auto& constraints = ctx->nd_sbp_constraints();
  if (constraints.bn_in_op2nd_sbp_size() != 0) {
    const auto it =
        constraints.bn_in_op2nd_sbp().find(GenRepeatedBn(in_arg.name(), in_arg.index()));
    if (it != constraints.bn_in_op2nd_sbp().end()) { *distribution = it->second; }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ApplySbpParallel(const SbpParallel& sbp, const int64_t parallel_num, Shape* shape) {
  if (sbp.has_split_parallel()) {
    const int64_t axis = sbp.split_parallel().axis();
    CHECK_EQ_OR_RETURN(shape->At(axis) % parallel_num, 0)
        << Error::RuntimeError() << "The size of tensor in the " << axis
        << " must be an integer multiple of parallel_num, "
        << "but got " << shape->At(axis) << " and " << parallel_num;
    shape->Set(axis, shape->At(axis) / parallel_num);
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> ReshapeUserOpUtil::InferNdSbp(user_op::InferNdSbpFnContext* ctx,
                                          const Shape& logical_in_shape,
                                          const Shape& logical_out_shape) {
  const std::string& op_type_name = ctx->user_op_conf().op_type_name();
  CHECK_OR_RETURN(op_type_name == "reshape" || op_type_name == "reshape_like")
      << Error::RuntimeError() << "The op_type_name must be \"reshape\" or \"reshape_like\", "
      << "but got " << op_type_name;
  const bool is_reshape_like = (op_type_name == "reshape_like");
  std::vector<user_op::OpArg> in_args({{"in", 0}});
  if (is_reshape_like) { in_args.emplace_back(user_op::OpArg("like", 0)); }
  HashMap<std::string, NdSbp> ibn2nd_sbp;
  ibn2nd_sbp.reserve(in_args.size());
  for (const auto& arg : in_args) {
    NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex(arg.name(), arg.index());
    JUST(GetInputNdSbp(ctx, arg, in_distribution));
    CHECK_OR_RETURN(
        ibn2nd_sbp.emplace(GenRepeatedBn(arg.name(), arg.index()), *in_distribution).second)
        << "emplace error";  // NOLINT(maybe-need-error-msg)
  }
  NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);

  Shape in_shape = logical_in_shape;
  Shape out_shape = logical_out_shape;
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  for (int64_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    SbpSignatureList sbp_sig_list;
    user_op::UserOpSbpSignatureBuilder builder(&sbp_sig_list);
    builder.Broadcast(in_args).Broadcast(user_op::OpArg("out", 0)).Build();
    if (is_reshape_like) {
      builder.PartialSum(user_op::OpArg("like", 0))
          .Broadcast(user_op::OpArg("in", 0))
          .Broadcast(user_op::OpArg("out", 0))
          .Build();
      builder.Broadcast(user_op::OpArg("like", 0))
          .PartialSum(user_op::OpArg("in", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
      JUST(GetReshapeUserOpSbpSignatures(in_shape, out_shape, {{"in", 0}},
                                         {{"like", 0}, {"out", 0}}, parallel_hierarchy.At(i),
                                         &builder));
    } else {
      JUST(GetReshapeUserOpSbpSignatures(in_shape, out_shape, {{"in", 0}}, {{"out", 0}},
                                         parallel_hierarchy.At(i), &builder));
    }

    const SbpSignature* matched_sbp_signature = nullptr;
    for (const auto& sbp_signature : sbp_sig_list.sbp_signature()) {
      bool all_match = true;
      for (const auto& in_arg : in_args) {
        std::string ibn = GenRepeatedBn(in_arg.name(), in_arg.index());
        if (sbp_signature.bn_in_op2sbp_parallel().at(ibn) != ibn2nd_sbp.at(ibn).sbp_parallel(i)) {
          all_match = false;
          break;
        }
      }
      if (all_match) {
        matched_sbp_signature = &sbp_signature;
        break;
      }
    }
    CHECK_OR_RETURN(matched_sbp_signature != nullptr)
        << "FusedLstmCellGrad::Pointer to the matched sbp signature is nullptr";
    SbpParallel out_sbp = matched_sbp_signature->bn_in_op2sbp_parallel().at("out_0");
    JUST(ApplySbpParallel(matched_sbp_signature->bn_in_op2sbp_parallel().at("in_0"),
                          parallel_hierarchy.At(i), &in_shape));
    JUST(ApplySbpParallel(out_sbp, parallel_hierarchy.At(i), &out_shape));
    *(out_distribution->add_sbp_parallel()) = out_sbp;
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
