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
#include "oneflow/core/common/container_util.h"

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
    const Shape& in_shape, const Shape& out_shape, const std::vector<user_op::OpArg>& in_args,
    const std::vector<user_op::OpArg>& out_args, const int64_t hierarchy_value,
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

namespace {

void FowardRankMesh(size_t depth, size_t max_depth, std::deque<int>& rank_axes_queue,
                    std::vector<std::vector<int>>& rank_axes_subset) {
  if (depth == max_depth) {
    // skip empty subset
    if (rank_axes_queue.empty()) { return; }
    rank_axes_subset.emplace_back();
    auto& rank_axes = rank_axes_subset.back();
    for (int rank_axis : rank_axes_queue) { rank_axes.push_back(rank_axis); }
  } else {
    // forward by skip current depth axis
    FowardRankMesh(depth + 1, max_depth, rank_axes_queue, rank_axes_subset);
    // fowward by keep current depth axis
    rank_axes_queue.push_back(depth);
    FowardRankMesh(depth + 1, max_depth, rank_axes_queue, rank_axes_subset);
    rank_axes_queue.pop_back();
  }
}

void GenRankMeshSubset(size_t mesh_depth, std::vector<std::vector<int>>& rank_axes_subset) {
  std::deque<int> rank_axes_queue;
  FowardRankMesh(0, mesh_depth, rank_axes_queue, rank_axes_subset);
}

}  // namespace

Maybe<void> ReshapeUserOpUtil::EnumerateNdSplitIn2OutAxis(
    const Shape& in_shape, const std::vector<int>& origin_in_axes, const Shape& out_shape,
    const std::vector<int>& origin_out_axes, const Shape& rank_mesh,
    std::vector<std::map<int, std::pair<int, int>>>* nd_split_groups) {
  CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), out_shape.elem_cnt());
  CHECK_EQ_OR_RETURN(in_shape.size(), origin_in_axes.size());
  CHECK_EQ_OR_RETURN(out_shape.size(), origin_out_axes.size());
  // generate all subset of rank_mesh (keep order)
  // for example rank_mesh=(2, 3, 5), subset include:
  // (2, 3, 5)
  // (2, 3)
  // (2, 5)
  // (2,)
  // (3, 5)
  // (3,)
  // (5,)
  std::vector<std::vector<int>> rank_axes_subset;
  GenRankMeshSubset(rank_mesh.size(), rank_axes_subset);
  // traverse all subset to detect contiguous nd-split signatures
  // for example (6,) reshape to (2, 3) with rank_mesh=(2, 3)
  // nd-split signatures include:
  // S(0) -> S(0) with rank_axis=0 (1d)
  // S(0) -> S(1) with rank_axis=1 (1d)
  // [S(0), S(0)] -> [S(0), S(1)] with rank_mesh=(2,3) (2d)
  for (const std::vector<int>& rank_axes : rank_axes_subset) {
    int rank_axis_idx = 0;
    int in_axis = in_shape.size() - 1;
    int out_axis = out_shape.size() - 1;
    int64_t in_dim_size = in_shape[in_axis];
    int64_t out_dim_size = out_shape[out_axis];
    // rank_axis -> {in_axis, out_axis}
    std::map<int, std::pair<int, int>> rank_in2out_axis;
    // go down from tail to head axis, since the dimensions
    // in the in_shape and the out_shape passed in
    // are reverse order
    while (in_axis >= 0 && out_axis >= 0 && rank_axis_idx < rank_axes.size()) {
      // dim_size == 1 then move to next axis to find contiguous split axis
      if (in_dim_size == 1) {
        in_axis--;
        in_dim_size = in_shape[in_axis];
        continue;
      }
      if (out_dim_size == 1) {
        out_axis--;
        out_dim_size = out_shape[out_axis];
        continue;
      }
      int rank_axis = rank_axes[rank_axis_idx];
      int64_t rank_num = rank_mesh[rank_axis];
      // dim_size is indivisible by rank_num indicate split can't continue
      if (in_dim_size % rank_num != 0 || out_dim_size % rank_num != 0) { break; }
      // divide dim_size by rank_num both at in_axis and out_axis till dim_size == 1
      in_dim_size /= rank_num;
      out_dim_size /= rank_num;
      int origin_in_axis = origin_in_axes[in_axis];
      int origin_out_axis = origin_out_axes[out_axis];
      // mark rank_axis that can be splited by in_axis and out_axis both
      rank_in2out_axis.emplace(rank_axis, std::make_pair(origin_in_axis, origin_out_axis));
      rank_axis_idx++;
    }
    // ensure all rank axes are marked splitable with some axis (in and out)
    if (rank_in2out_axis.size() == rank_axes.size()) {
      nd_split_groups->emplace_back(std::move(rank_in2out_axis));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::EnumerateNdSplitIn2OutAxisGroups(
    const Shape& in_shape, const Shape& out_shape, const Shape& rank_mesh,
    std::vector<std::map<int, std::pair<int, int>>>* nd_sbp_in2out_sig_groups) {
  int in_axis = in_shape.size();
  int out_axis = out_shape.size();
  int64_t in_count = 1;
  int64_t out_count = 1;
  auto MoveAxis = [](const Shape& shape, int& axis, int64_t& count) {
    axis--;
    if (axis >= 0 && axis < shape.size()) { count *= shape[axis]; }
  };
  auto MoveInAxis = [&]() { MoveAxis(in_shape, in_axis, in_count); };
  auto MoveOutAxis = [&]() { MoveAxis(out_shape, out_axis, out_count); };
  MoveInAxis();
  MoveOutAxis();

  DimVector group_in_dim_vec;
  DimVector group_out_dim_vec;
  std::vector<int> group_in_axes;
  std::vector<int> group_out_axes;
  group_in_axes.reserve(rank_mesh.size());
  group_out_axes.reserve(rank_mesh.size());

  // group reshape dimensions
  // for example:
  // (4, 5, 2, 3) reshape to (2, 2, 5, 6) will be divided to 3 groups:
  // (   4,| 5, | 2, 3)
  // (2, 2,| 5, | 6)
  // group1: (2, 3) -> (6)
  // group2: (5,) -> (5)
  // group3: (4,) -> (2, 2)
  while (in_axis >= 0 && out_axis >= 0) {
    // move in_axis when in_count < out_count
    // move out_axis when out_count < in_count
    // move both when in_count == out_count
    if (in_count < out_count) {
      // skip dim_size == 1
      if (in_shape[in_axis] != 1) {
        group_in_dim_vec.push_back(in_shape[in_axis]);
        group_in_axes.push_back(in_axis);
      }
      MoveInAxis();
    } else if (in_count > out_count) {
      if (out_shape[out_axis] != 1) {
        group_out_dim_vec.push_back(out_shape[out_axis]);
        group_out_axes.push_back(out_axis);
      }
      MoveOutAxis();
    } else {  // in_count == out_count
      if (in_shape[in_axis] == out_shape[out_axis]) {
        // group2: (5, 5) in the example will reach this branch
        for (int rank_axis = 0; rank_axis < rank_mesh.size(); ++rank_axis) {
          int64_t rank_num = rank_mesh[rank_axis];
          if (in_shape[in_axis] % rank_num == 0) {
            std::map<int, std::pair<int, int>> rank_in2out_split_axis{
                {rank_axis, std::make_pair(in_axis, out_axis)}};
            nd_sbp_in2out_sig_groups->emplace_back(std::move(rank_in2out_split_axis));
          }
        }
      } else {
        // the reshape group (group1 and group3 in the example) finish
        group_in_dim_vec.push_back(in_shape[in_axis]);
        group_in_axes.push_back(in_axis);
        group_out_dim_vec.push_back(out_shape[out_axis]);
        group_out_axes.push_back(out_axis);
        // enumerate all nd-split signatures for one group
        JUST(EnumerateNdSplitIn2OutAxis(Shape(group_in_dim_vec), group_in_axes,
                                        Shape(group_out_dim_vec), group_out_axes, rank_mesh,
                                        nd_sbp_in2out_sig_groups));
        group_in_dim_vec.clear();
        group_out_dim_vec.clear();
        group_in_axes.clear();
        group_out_axes.clear();
      }
      MoveInAxis();
      MoveOutAxis();
    }
  }

  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::DfsCombineNdSbpSignatureGroups(
    const std::vector<std::map<int, std::pair<int, int>>>& nd_sbp_sig_groups, size_t rank_num_axes,
    std::vector<std::vector<std::pair<int, int>>>* nd_sbp_sig_list) {
  std::map<int, std::pair<int, int>> nd_sbp_sig_group;
  std::set<std::vector<std::pair<int, int>>> nd_sbp_sig_set;
  JUST(DfsCombineNdSbpSignatureGroups(nd_sbp_sig_groups, rank_num_axes, nd_sbp_sig_group,
                                      nd_sbp_sig_set));
  std::copy(nd_sbp_sig_set.begin(), nd_sbp_sig_set.end(), back_inserter(*nd_sbp_sig_list));
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::DfsCombineNdSbpSignatureGroups(
    const std::vector<std::map<int, std::pair<int, int>>>& nd_sbp_sig_groups, size_t rank_num_axes,
    const std::map<int, std::pair<int, int>>& nd_sbp_sig_group,
    std::set<std::vector<std::pair<int, int>>>& nd_sbp_sig_set) {
  if (nd_sbp_sig_group.size() == rank_num_axes) {
    std::vector<std::pair<int, int>> nd_sbp_sig;
    for (int i = 0; i < rank_num_axes; ++i) {
      nd_sbp_sig.emplace_back(JUST(MapAt(nd_sbp_sig_group, i)));
    }
    nd_sbp_sig_set.emplace(nd_sbp_sig);
  } else {
    for (const auto& nd_sbp_sig_group_to_combine : nd_sbp_sig_groups) {
      std::map<int, std::pair<int, int>> new_nd_sbp_sig_group = nd_sbp_sig_group;
      bool combine_failed = false;
      for (const auto& rank_in2out_pair : nd_sbp_sig_group_to_combine) {
        int rank_axis = rank_in2out_pair.first;
        if (nd_sbp_sig_group.find(rank_axis) != nd_sbp_sig_group.end()) {
          combine_failed = true;
          break;
        }
        CHECK_OR_RETURN(new_nd_sbp_sig_group.emplace(rank_axis, rank_in2out_pair.second).second);
      }
      if (!combine_failed) {
        JUST(DfsCombineNdSbpSignatureGroups(nd_sbp_sig_groups, rank_num_axes, new_nd_sbp_sig_group,
                                            nd_sbp_sig_set));
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::EnumerateNdSbpIn2OutSignatures(
    const Shape& in_shape, const Shape& out_shape, const Shape& rank_mesh,
    std::vector<std::vector<std::pair<int, int>>>* nd_sbp_in2out_signatures) {
  CHECK_GT_OR_RETURN(in_shape.size(), 0)
      << Error::RuntimeError() << "The dimension of input tensor must be greater than zero, "
      << "but got " << in_shape.size();
  CHECK_GT_OR_RETURN(out_shape.size(), 0)
      << Error::RuntimeError() << "The dimension of output tensor must be greater than zero, "
      << "but got " << out_shape.size();
  CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), out_shape.elem_cnt())
      << Error::RuntimeError()
      << "The element number of input tensor must be equal to output tensor, "
      << "but got " << in_shape.elem_cnt() << " and " << out_shape.elem_cnt();

  // groups of nd of rank_axis -> (in_axis, out_axis)
  std::vector<std::map<int, std::pair<int, int>>> nd_sbp_signature_groups;
  JUST(EnumerateNdSplitIn2OutAxisGroups(in_shape, out_shape, rank_mesh, &nd_sbp_signature_groups));

  std::map<int, std::pair<int, int>> nd_sbp_in2out_group;
  for (int rank_axis = 0; rank_axis < rank_mesh.size(); ++rank_axis) {
    // -1 indicate broadcaste, -2 indicate partial sum
    nd_sbp_in2out_group.emplace(rank_axis, std::make_pair(-1, -1));
    nd_sbp_signature_groups.emplace_back(nd_sbp_in2out_group);
    nd_sbp_in2out_group.clear();
    nd_sbp_in2out_group.emplace(rank_axis, std::make_pair(-2, -2));
    nd_sbp_signature_groups.emplace_back(nd_sbp_in2out_group);
    nd_sbp_in2out_group.clear();
  }

  JUST(DfsCombineNdSbpSignatureGroups(nd_sbp_signature_groups, rank_mesh.size(),
                                      nd_sbp_in2out_signatures));
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::FilterNdSbpIn2OutSignatures(
    const Shape& in_shape, const Shape& out_shape, const Shape& rank_mesh,
    std::vector<std::vector<std::pair<int, int>>>* nd_sbp_in2out_signatures) {
  // filter the Nd SBP candidates
  // Go down from the tail to the head, since we might drop the tail.
  for (int i = nd_sbp_in2out_signatures->size() - 1; i >= 0; --i) {
    auto& nd_sbp_sig = (*nd_sbp_in2out_signatures)[i];
    CHECK_EQ_OR_RETURN(nd_sbp_sig.size(), rank_mesh.size());
    bool match_failed = false;
    DimVector in_dim_vec = in_shape.dim_vec();
    DimVector out_dim_vec = out_shape.dim_vec();
    for (int rank_axis = 0; rank_axis < nd_sbp_sig.size(); ++rank_axis) {
      int64_t rank_num = rank_mesh[rank_axis];
      int in_sig = nd_sbp_sig[rank_axis].first;
      int out_sig = nd_sbp_sig[rank_axis].second;
      if (in_sig >= 0) {
        if (in_dim_vec[in_sig] % rank_num == 0) {
          in_dim_vec[in_sig] /= rank_num;
        } else {
          match_failed = true;
          break;
        }
      }
      if (out_sig >= 0) {
        if (out_dim_vec[out_sig] % rank_num == 0) {
          out_dim_vec[out_sig] /= rank_num;
        } else {
          match_failed = true;
          break;
        }
      }
    }
    if (match_failed) {
      // swap the invalid Nd SBP with the tail and drop it
      std::swap(nd_sbp_sig, nd_sbp_in2out_signatures->back());
      nd_sbp_in2out_signatures->pop_back();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeUserOpUtil::EnumerateNdSbpSignatures(
    const std::vector<user_op::OpArg>& in_args, const Shape& in_shape,
    const std::vector<user_op::OpArg>& out_args, const Shape& out_shape, const Shape& rank_mesh,
    std::vector<NdSbpSignature>* nd_sbp_sig_list) {
  CHECK_EQ_OR_RETURN(in_shape.elem_cnt(), out_shape.elem_cnt());
  if (in_shape.elem_cnt() == 0) { return Maybe<void>::Ok(); }
  if (in_shape.size() == 0 || out_shape.size() == 0) { return Maybe<void>::Ok(); }
  std::vector<std::vector<std::pair<int, int>>> nd_sbp_in2out_sig_list;
  JUST(EnumerateNdSbpIn2OutSignatures(in_shape, out_shape, rank_mesh, &nd_sbp_in2out_sig_list));
  for (const auto& nd_sbp_in2out_axis : nd_sbp_in2out_sig_list) {
    nd_sbp_sig_list->emplace_back();
    auto& nd_sbp_sig = nd_sbp_sig_list->back();
    for (const auto& in2out_axis : nd_sbp_in2out_axis) {
      for (const auto& in_arg : in_args) {
        const auto& ibn = in_arg.name() + "_" + std::to_string(in_arg.index());
        auto& in_nd_sbp = (*nd_sbp_sig.mutable_bn_in_op2nd_sbp())[ibn];
        auto* in_sbp = in_nd_sbp.add_sbp_parallel();
        if (in2out_axis.first == -1) {
          in_sbp->mutable_broadcast_parallel();
        } else if (in2out_axis.first == -2) {
          in_sbp->mutable_partial_sum_parallel();
        } else {
          in_sbp->mutable_split_parallel()->set_axis(in2out_axis.first);
        }
      }
      for (const auto& out_arg : out_args) {
        const auto& obn = out_arg.name() + "_" + std::to_string(out_arg.index());
        auto& out_nd_sbp = (*nd_sbp_sig.mutable_bn_in_op2nd_sbp())[obn];
        auto* out_sbp = out_nd_sbp.add_sbp_parallel();
        if (in2out_axis.second == -1) {
          out_sbp->mutable_broadcast_parallel();
        } else if (in2out_axis.second == -2) {
          out_sbp->mutable_partial_sum_parallel();
        } else {
          out_sbp->mutable_split_parallel()->set_axis(in2out_axis.second);
        }
      }
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
