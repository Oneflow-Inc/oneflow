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
#include <tuple>
#include <algorithm>
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/common/tensor_meta.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/math_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace private_details {

namespace {

using IndexVector = DimVector;

Maybe<void> GetIndexesFromOffset(const Stride& strides, int64_t offset, IndexVector* indexes) {
  indexes->resize(strides.size());
  for (int i = 0; i < strides.size(); ++i) {
    indexes->at(i) = offset / strides.at(i);
    offset = offset % strides.at(i);
  }
  CHECK_EQ_OR_RETURN(offset, 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetOffsetFromIndexes(const Stride& strides, const IndexVector& indexes,
                                 int64_t* offset) {
  CHECK_EQ_OR_RETURN(strides.size(), indexes.size())
      << Error::RuntimeError() << "Expected size of strides to match that of indexes";
  *offset = 0;
  for (int i = 0; i < strides.size(); ++i) { *offset += indexes.at(i) * strides.at(i); }
  return Maybe<void>::Ok();
}

Maybe<void> GetSelectedIndex2OriginIndex(
    const IndexVector& indexes, const std::vector<int>& axis2is_selected,
    std::function<void(const DimVector&, DimVector*)>* SelectedIndex2OriginIndex) {
  CHECK_EQ_OR_RETURN(axis2is_selected.size(), indexes.size());
  *SelectedIndex2OriginIndex = [=](const DimVector& broadcast, DimVector* origin) {
    origin->resize(indexes.size());
    for (int i = 0; i < indexes.size(); ++i) {
      origin->at(i) = axis2is_selected.at(i) ? broadcast.at(i) : indexes.at(i);
    }
  };
  return Maybe<void>::Ok();
}

Maybe<const Shape> GetSelectedShape(const Shape& hierarchy_shape,
                                    const std::vector<int>& axis2is_selected) {
  CHECK_EQ_OR_RETURN(hierarchy_shape.NumAxes(), axis2is_selected.size());
  DimVector dim_vec = hierarchy_shape.dim_vec();
  for (int i = 0; i < axis2is_selected.size(); ++i) {
    if (!axis2is_selected.at(i)) { dim_vec.at(i) = 1; }
  }
  return std::make_shared<const Shape>(dim_vec);
}

Maybe<Symbol<std::vector<int>>> CalcAxis2IsBroadcast(Symbol<NdSbp> nd_sbp) {
  std::vector<int> axis2is_selected(nd_sbp->sbp_parallel_size());
  for (int i = 0; i < axis2is_selected.size(); ++i) {
    axis2is_selected.at(i) = nd_sbp->sbp_parallel(i).has_broadcast_parallel();
  }
  return SymbolOf(axis2is_selected);
}

static auto* GetAxis2IsBroadcast = DECORATE(&CalcAxis2IsBroadcast, ThreadLocal);

Maybe<Symbol<ParallelDesc>> CalcSelectedSubParallelDesc(Symbol<ParallelDesc> parallel_desc,
                                                        Symbol<std::vector<int>> axis2is_selected) {
  const auto& opt_parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
  int64_t parallel_id = JUST(*opt_parallel_id);
  const auto& hierarchy_shape = *parallel_desc->hierarchy();
  const auto& broadcast_parallel_ids =
      JUST(GetSelectedParallelIds(hierarchy_shape, *axis2is_selected, parallel_id));
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(parallel_desc->device_tag());
  bool found_parallel_id = false;
  for (int64_t i : *broadcast_parallel_ids) {
    found_parallel_id = found_parallel_id || (i == parallel_id);
    int64_t machine_id = JUST(parallel_desc->MachineId4ParallelId(i));
    int64_t device_id = JUST(parallel_desc->DeviceId4ParallelId(i));
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":"
                                  + std::to_string(device_id));
  }
  CHECK_OR_RETURN(found_parallel_id);
  return SymbolOf(ParallelDesc(parallel_conf));
}

static auto* GetSelectedSubParallelDesc = DECORATE(&CalcSelectedSubParallelDesc, ThreadLocal);

}  // namespace

Maybe<Symbol<ParallelDesc>> CalcSubParallelDesc4Axis(Symbol<ParallelDesc> parallel_desc, int axis) {
  const auto& opt_parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
  int64_t parallel_id = JUST(*opt_parallel_id);
  const auto& hierarchy_shape = *parallel_desc->hierarchy();
  Stride hierarchy_strides(hierarchy_shape);

  int64_t index = CalcIndex4Axis(parallel_id, hierarchy_strides, axis);

  int64_t stride = hierarchy_strides.at(axis);

  int64_t start_parallel_id = parallel_id - index * stride;
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(parallel_desc->device_tag());
  for (int64_t i = 0; i < hierarchy_shape.At(axis); ++i) {
    int64_t id = start_parallel_id + i * stride;
    int64_t machine_id = JUST(parallel_desc->MachineId4ParallelId(id));
    int64_t device_id = JUST(parallel_desc->DeviceId4ParallelId(id));
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":"
                                  + std::to_string(device_id));
  }
  return SymbolOf(ParallelDesc(parallel_conf));
}

Maybe<std::vector<int64_t>> GetSelectedParallelIds(const Shape& hierarchy_shape,
                                                   const std::vector<int>& axis2is_selected,
                                                   int64_t parallel_id) {
  CHECK_EQ_OR_RETURN(hierarchy_shape.NumAxes(), axis2is_selected.size());
  Stride hierarchy_strides(hierarchy_shape);
  IndexVector indexes{};
  JUST(GetIndexesFromOffset(hierarchy_strides, parallel_id, &indexes));
  std::function<void(const DimVector&, DimVector*)> SelectedIndex2OriginIndex;
  JUST(GetSelectedIndex2OriginIndex(indexes, axis2is_selected, &SelectedIndex2OriginIndex));
  const auto& broadcast_shape = JUST(GetSelectedShape(hierarchy_shape, axis2is_selected));
  Stride broadcast_strides(*broadcast_shape);
  const auto& origin_offsets = std::make_shared<std::vector<int64_t>>(broadcast_shape->elem_cnt());
  for (int64_t i = 0; i < broadcast_shape->elem_cnt(); ++i) {
    IndexVector broadcast_indexes{};
    JUST(GetIndexesFromOffset(broadcast_strides, i, &broadcast_indexes));
    IndexVector origin_indexes{};
    SelectedIndex2OriginIndex(broadcast_indexes, &origin_indexes);
    int64_t origin_offset = -1;
    JUST(GetOffsetFromIndexes(hierarchy_strides, origin_indexes, &origin_offset));
    origin_offsets->at(i) = origin_offset;
  }
  return origin_offsets;
}

Maybe<Symbol<ParallelDesc>> GetBroadcastSubParallelDesc(Symbol<ParallelDesc> parallel_desc,
                                                        Symbol<NdSbp> nd_sbp) {
  const auto& axis2is_selected = JUST(GetAxis2IsBroadcast(nd_sbp));
  return GetSelectedSubParallelDesc(parallel_desc, axis2is_selected);
}

namespace {

Maybe<Symbol<NdSbp>> MakeNdSbp(const SbpParallel& sbp) {
  NdSbp nd_sbp;
  nd_sbp.mutable_sbp_parallel()->Add()->CopyFrom(sbp);
  return SymbolOf(nd_sbp);
}

Maybe<void> InitShapeAxis2NdSbpIndexes(
    Symbol<NdSbp> nd_sbp, std::vector<std::vector<int64_t>>* shape_axis2nd_sbp_indexes) {
  for (int i = 0; i < nd_sbp->sbp_parallel_size(); ++i) {
    const auto& sbp = nd_sbp->sbp_parallel(i);
    if (sbp.has_split_parallel()) {
      int64_t axis = sbp.split_parallel().axis();
      CHECK_GE_OR_RETURN(axis, 0);
      CHECK_LT_OR_RETURN(axis, shape_axis2nd_sbp_indexes->size());
      shape_axis2nd_sbp_indexes->at(axis).emplace_back(i);
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> CheckSplitAxisExpandable(
    const Shape& hierarchy, const std::vector<std::vector<int64_t>>& shape_axis2src_nd_sbp_indexes,
    const std::vector<std::vector<int64_t>>& shape_axis2dst_nd_sbp_indexes) {
  const auto& GetHierarchyDim = [&](int64_t axis) { return hierarchy.At(axis); };
  for (int i = 0; i < shape_axis2src_nd_sbp_indexes.size(); ++i) {
    const auto& src_nd_sbp_indexes = JUST(VectorAt(shape_axis2src_nd_sbp_indexes, i));
    if (src_nd_sbp_indexes.empty()) { continue; }
    const auto& dst_nd_sbp_indexes = JUST(VectorAt(shape_axis2dst_nd_sbp_indexes, i));
    if (dst_nd_sbp_indexes.empty()) { continue; }
    std::vector<int64_t> src_nd_sbp_dims{};
    src_nd_sbp_dims.reserve(src_nd_sbp_indexes.size());
    std::transform(src_nd_sbp_indexes.begin(), src_nd_sbp_indexes.end(),
                   std::back_inserter(src_nd_sbp_dims), GetHierarchyDim);
    std::vector<int64_t> dst_nd_sbp_dims{};
    dst_nd_sbp_dims.reserve(dst_nd_sbp_indexes.size());
    std::transform(dst_nd_sbp_indexes.begin(), dst_nd_sbp_indexes.end(),
                   std::back_inserter(dst_nd_sbp_dims), GetHierarchyDim);
    CHECK_OR_RETURN(src_nd_sbp_dims == dst_nd_sbp_dims) << Error::BoxingNotSupportedError();
  }
  return Maybe<void>::Ok();
}

Maybe<void> InitShapAxis2ExpandedDim(
    std::vector<DimVector>* shape_axis2expanded_dims, const Shape& shape, const Shape& hierarchy,
    const std::vector<std::vector<int64_t>>& shape_axis2src_nd_sbp_indexes,
    const std::vector<std::vector<int64_t>>& shape_axis2dst_nd_sbp_indexes) {
  std::vector<DimVector> shape_axis2required_dim(shape.NumAxes());
  for (int i = 0; i < shape.NumAxes(); ++i) {
    const auto& src_nd_sbp_indexes = shape_axis2src_nd_sbp_indexes.at(i);
    const auto& dst_nd_sbp_indexes = shape_axis2dst_nd_sbp_indexes.at(i);
    int64_t max_used_cnt = std::max<size_t>(src_nd_sbp_indexes.size(), dst_nd_sbp_indexes.size());
    for (int j = 0; j < max_used_cnt; ++j) {
      if (j < src_nd_sbp_indexes.size() && j < dst_nd_sbp_indexes.size()) {
        int64_t m = hierarchy.At(src_nd_sbp_indexes.at(j));
        int64_t n = hierarchy.At(dst_nd_sbp_indexes.at(j));
        shape_axis2required_dim.at(i).emplace_back(Lcm(m, n));
      } else if (j < src_nd_sbp_indexes.size()) {
        shape_axis2required_dim.at(i).emplace_back(hierarchy.At(src_nd_sbp_indexes.at(j)));
      } else if (j < dst_nd_sbp_indexes.size()) {
        shape_axis2required_dim.at(i).emplace_back(hierarchy.At(dst_nd_sbp_indexes.at(j)));
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    }
  }
  for (int i = 0; i < shape.NumAxes(); ++i) {
    int64_t total_dim = shape.At(i);
    shape_axis2expanded_dims->at(i).clear();
    if (JUST(VectorAt(shape_axis2required_dim, i)).empty()
        || JUST(VectorAt(shape_axis2required_dim, i)).size() == 1) {
      shape_axis2expanded_dims->at(i).emplace_back(total_dim);
    } else {
      Shape inner_shape(shape_axis2required_dim.at(i));
      CHECK_EQ_OR_RETURN(total_dim % inner_shape.elem_cnt(), 0)
          << "dim " << total_dim << "(axis " << i << " in shape " << shape.ToString() << ")"
          << " cannot be reshape into exapanded shape " << inner_shape.ToString();
      auto* dim_vec = &shape_axis2expanded_dims->at(i);
      *dim_vec = shape_axis2required_dim.at(i);
      dim_vec->at(dim_vec->size() - 1) *= total_dim / inner_shape.elem_cnt();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<const Shape> Flatten(const std::vector<DimVector>& shape_axis2expanded_dims) {
  DimVector dim_vec;
  for (const auto& expanded_dims : shape_axis2expanded_dims) {
    CHECK_OR_RETURN(!expanded_dims.empty());
    dim_vec.insert(dim_vec.end(), expanded_dims.begin(), expanded_dims.end());
  }
  return std::make_shared<const Shape>(dim_vec);
}

Maybe<void> InitOldAxis2NewAxisOffset(std::vector<int64_t>* old_axis2new_axis_offset,
                                      const std::vector<DimVector>& shape_axis2expanded_dims) {
  for (int i = 0, offset = 0; i < shape_axis2expanded_dims.size(); ++i) {
    old_axis2new_axis_offset->at(i) = offset;
    offset += shape_axis2expanded_dims.at(i).size();
  }
  return Maybe<void>::Ok();
}

Maybe<Symbol<NdSbp>> ShiftSplitAxis(
    Symbol<NdSbp> nd_sbp, const std::vector<std::vector<int64_t>>& shape_axis2nd_sbp_indexes,
    const std::vector<int64_t>& old_axis2new_axis_offset) {
  CHECK_EQ_OR_RETURN(shape_axis2nd_sbp_indexes.size(), old_axis2new_axis_offset.size());
  NdSbp new_nd_sbp(*nd_sbp);
  for (int axis = 0; axis < shape_axis2nd_sbp_indexes.size(); ++axis) {
    int64_t offset = old_axis2new_axis_offset.at(axis);
    for (int64_t j = 0; j < shape_axis2nd_sbp_indexes.at(axis).size(); ++j) {
      int64_t nd_sbp_index = shape_axis2nd_sbp_indexes.at(axis).at(j);
      CHECK_GE_OR_RETURN(nd_sbp_index, 0);
      CHECK_LT_OR_RETURN(nd_sbp_index, new_nd_sbp.sbp_parallel_size());
      auto* sbp_parallel = new_nd_sbp.mutable_sbp_parallel(nd_sbp_index);
      CHECK_OR_RETURN(sbp_parallel->has_split_parallel());
      CHECK_EQ_OR_RETURN(sbp_parallel->split_parallel().axis(), axis);
      sbp_parallel->mutable_split_parallel()->set_axis(offset + j);
    }
  }
  return SymbolOf(new_nd_sbp);
}

}  // namespace

Maybe<std::tuple<std::shared_ptr<const Shape>, Symbol<NdSbp>, Symbol<NdSbp>>>
CalcDecomposableEquivalentShapeAndNdSbpPair(const Shape& shape, const Shape& hierarchy,
                                            Symbol<NdSbp> src_nd_sbp, Symbol<NdSbp> dst_nd_sbp) {
  CHECK_EQ_OR_RETURN(src_nd_sbp->sbp_parallel_size(), dst_nd_sbp->sbp_parallel_size());
  std::vector<std::vector<int64_t>> shape_axis2src_nd_sbp_indexes(shape.NumAxes());
  JUST(InitShapeAxis2NdSbpIndexes(src_nd_sbp, &shape_axis2src_nd_sbp_indexes));
  std::vector<std::vector<int64_t>> shape_axis2dst_nd_sbp_indexes(shape.NumAxes());
  JUST(InitShapeAxis2NdSbpIndexes(dst_nd_sbp, &shape_axis2dst_nd_sbp_indexes));
  std::vector<DimVector> shape_axis2expanded_dims(shape.NumAxes());
  CHECK_EQ_OR_RETURN(hierarchy.NumAxes(), src_nd_sbp->sbp_parallel_size());
  JUST(CheckSplitAxisExpandable(hierarchy, shape_axis2src_nd_sbp_indexes,
                                shape_axis2dst_nd_sbp_indexes));
  JUST(InitShapAxis2ExpandedDim(&shape_axis2expanded_dims, shape, hierarchy,
                                shape_axis2src_nd_sbp_indexes, shape_axis2dst_nd_sbp_indexes));
  std::shared_ptr<const Shape> new_shape = JUST(Flatten(shape_axis2expanded_dims));
  CHECK_EQ_OR_RETURN(new_shape->elem_cnt(), shape.elem_cnt());
  std::vector<int64_t> old_axis2new_axis_offset(shape.NumAxes());
  JUST(InitOldAxis2NewAxisOffset(&old_axis2new_axis_offset, shape_axis2expanded_dims));
  Symbol<NdSbp> new_src_nd_sbp =
      JUST(ShiftSplitAxis(src_nd_sbp, shape_axis2src_nd_sbp_indexes, old_axis2new_axis_offset));
  Symbol<NdSbp> new_dst_nd_sbp =
      JUST(ShiftSplitAxis(dst_nd_sbp, shape_axis2dst_nd_sbp_indexes, old_axis2new_axis_offset));
  return std::make_tuple(new_shape, new_src_nd_sbp, new_dst_nd_sbp);
}

namespace {

// nd_sbp is called decomposable if no particular axis is used to split tensor more than once.
// e.g.
// 1) (S0, S1) is decomposable.
// 2) (S0, S0) is not decomposable.
// 3) (S1, S1) is not decomposable.
// although `nd_sbp (S0, S0) on shape (4, 4)` is not decomposable, they could be transformed into a
// decomposable form: `n_sbp (S0, S1) on shape (2, 2, 4)`.
Maybe<std::pair<Symbol<one::GlobalTensorMeta>, Symbol<NdSbp>>> CalcDecomposableEquivalent(
    Symbol<one::GlobalTensorMeta> tensor_meta, Symbol<NdSbp> dst_nd_sbp) {
  std::shared_ptr<const Shape> shape = tensor_meta->shape_ptr();
  Symbol<NdSbp> src_nd_sbp = tensor_meta->nd_sbp();
  const auto& hierarchy = tensor_meta->parallel_desc()->hierarchy();
  std::tie(shape, src_nd_sbp, dst_nd_sbp) = *JUST(
      CalcDecomposableEquivalentShapeAndNdSbpPair(*shape, *hierarchy, src_nd_sbp, dst_nd_sbp));

  one::GlobalTensorMeta decomposible_tensor_meta(*shape, tensor_meta->dtype(), src_nd_sbp,
                                                 tensor_meta->parallel_desc());
  return std::make_pair(SymbolOf(decomposible_tensor_meta), dst_nd_sbp);
}

static constexpr auto* GetDecomposableEquivalent =
    DECORATE(&CalcDecomposableEquivalent, ThreadLocal);

Maybe<void> InitDstNdSbpAxis2ExclusiveSrcNdSbpAxis(
    HashMap<int64_t, int64_t>* dst_nd_sbp_axis2exclusive_src_nd_sbp_axis, Symbol<NdSbp> src_nd_sbp,
    Symbol<NdSbp> dst_nd_sbp) {
  HashMap<int64_t, int64_t> split_axis2src_nd_sbp_axis;
  for (int i = 0; i < src_nd_sbp->sbp_parallel_size(); ++i) {
    const auto& sbp_parallel = src_nd_sbp->sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      split_axis2src_nd_sbp_axis[sbp_parallel.split_parallel().axis()] = i;
    }
  }
  for (int i = 0; i < dst_nd_sbp->sbp_parallel_size(); ++i) {
    const auto& sbp_parallel = dst_nd_sbp->sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      int64_t axis = sbp_parallel.split_parallel().axis();
      const auto& iter = split_axis2src_nd_sbp_axis.find(axis);
      if (iter != split_axis2src_nd_sbp_axis.end() && iter->second != i) {
        (*dst_nd_sbp_axis2exclusive_src_nd_sbp_axis)[i] = iter->second;
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> MakeExclusiveSrcNdSbpAxis4DstNdSbpAxis(
    std::function<Maybe<Optional<int64_t>>(int64_t)>* ExclusiveSrcNdSbpAxis4DstNdSbpAxis,
    Symbol<NdSbp> src_nd_sbp, Symbol<NdSbp> dst_nd_sbp) {
  CHECK_EQ_OR_RETURN(src_nd_sbp->sbp_parallel_size(), dst_nd_sbp->sbp_parallel_size());
  HashMap<int64_t, int64_t> split_axis2src_nd_sbp_axis;
  for (int i = 0; i < src_nd_sbp->sbp_parallel_size(); ++i) {
    const auto& sbp_parallel = src_nd_sbp->sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      int64_t split_axis = sbp_parallel.split_parallel().axis();
      CHECK_OR_RETURN(split_axis2src_nd_sbp_axis.emplace(split_axis, i).second);
    }
  }
  {
    // check split_axis used only once.
    HashMap<int64_t, int64_t> split_axis2dst_nd_sbp_axis;
    for (int i = 0; i < dst_nd_sbp->sbp_parallel_size(); ++i) {
      const auto& sbp_parallel = dst_nd_sbp->sbp_parallel(i);
      if (sbp_parallel.has_split_parallel()) {
        int64_t split_axis = sbp_parallel.split_parallel().axis();
        CHECK_OR_RETURN(split_axis2dst_nd_sbp_axis.emplace(split_axis, i).second);
      }
    }
  }
  *ExclusiveSrcNdSbpAxis4DstNdSbpAxis = [split_axis2src_nd_sbp_axis, src_nd_sbp,
                                         dst_nd_sbp](int64_t dst_axis) -> Maybe<Optional<int64_t>> {
    CHECK_GE_OR_RETURN(dst_axis, 0);
    CHECK_LT_OR_RETURN(dst_axis, dst_nd_sbp->sbp_parallel_size());
    const auto& dst_sbp_parallel = dst_nd_sbp->sbp_parallel(dst_axis);
    if (!dst_sbp_parallel.has_split_parallel()) { return Optional<int64_t>(); }
    int64_t split_axis = dst_sbp_parallel.split_parallel().axis();
    const auto& src_iter = split_axis2src_nd_sbp_axis.find(split_axis);
    if (src_iter == split_axis2src_nd_sbp_axis.end()) { return Optional<int64_t>(); }
    int64_t src_axis = src_iter->second;
    CHECK_GE_OR_RETURN(src_axis, 0);
    CHECK_LT_OR_RETURN(src_axis, dst_nd_sbp->sbp_parallel_size());
    const auto& src_sbp_parallel = src_nd_sbp->sbp_parallel(src_axis);
    CHECK_OR_RETURN(src_sbp_parallel.has_split_parallel());
    CHECK_EQ_OR_RETURN(src_sbp_parallel.split_parallel().axis(), split_axis);
    if (src_axis == dst_axis) { return Optional<int64_t>(); }
    return Optional<int64_t>(src_axis);
  };
  return Maybe<void>::Ok();
}

Maybe<bool> IsNdSbpBoxingAcyclic(
    int64_t num_axes,
    const std::function<Maybe<Optional<int64_t>>(int64_t)>& ExclusiveSrcNdSbpAxis4DstNdSbpAxis) {
  for (int start_axis = 0; start_axis < num_axes; ++start_axis) {
    int64_t axis = start_axis;
    HashSet<int64_t> visited_axes;
    for (int i = 0; i < num_axes + 1; ++i) {
      const auto& opt_axis = JUST(ExclusiveSrcNdSbpAxis4DstNdSbpAxis(axis));
      if (!opt_axis->has_value()) { break; }
      axis = JUST(*opt_axis);
      if (!visited_axes.insert(axis).second) { return false; }
    }
  }
  return true;
}

Maybe<void> InitNdSbpValidTransformationAxisSequence(
    std::vector<int64_t>* nd_sbp_axis_sequence, Symbol<NdSbp> src_nd_sbp, Symbol<NdSbp> dst_nd_sbp,
    const std::function<Maybe<Optional<int64_t>>(int64_t)>& ExclusiveSrcNdSbpAxis4DstNdSbpAxis) {
  CHECK_EQ_OR_RETURN(src_nd_sbp->sbp_parallel_size(), dst_nd_sbp->sbp_parallel_size());
  int64_t num_axes = src_nd_sbp->sbp_parallel_size();
  HashSet<int64_t> handled_axes;
  nd_sbp_axis_sequence->reserve(num_axes);
  const auto& HasNoExclusiveSrcNdSbpAxis = [&](int64_t axis) -> Maybe<bool> {
    const auto& opt_src_axis = JUST(ExclusiveSrcNdSbpAxis4DstNdSbpAxis(axis));
    if (!opt_src_axis->has_value()) { return true; }
    return handled_axes.count(JUST(*opt_src_axis)) > 0;
  };
  for (int i = 0; i < num_axes; ++i) {
    for (int axis = 0; axis < num_axes; ++axis) {
      if (handled_axes.count(axis) == 0 && JUST(HasNoExclusiveSrcNdSbpAxis(axis))) {
        if (!(src_nd_sbp->sbp_parallel(axis) == dst_nd_sbp->sbp_parallel(axis))) {
          nd_sbp_axis_sequence->emplace_back(axis);
        }
        handled_axes.insert(axis);
      }
    }
  }
  CHECK_EQ_OR_RETURN(handled_axes.size(), num_axes);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<bool> IsNdSbpBoxingAcyclic(Symbol<NdSbp> src_nd_sbp, Symbol<NdSbp> dst_nd_sbp) {
  std::function<Maybe<Optional<int64_t>>(int64_t)> ExclusiveSrcNdSbpAxis4DstNdSbpAxis;
  JUST(MakeExclusiveSrcNdSbpAxis4DstNdSbpAxis(&ExclusiveSrcNdSbpAxis4DstNdSbpAxis, src_nd_sbp,
                                              dst_nd_sbp));
  return IsNdSbpBoxingAcyclic(src_nd_sbp->sbp_parallel_size(), ExclusiveSrcNdSbpAxis4DstNdSbpAxis);
}

Maybe<std::vector<int64_t>> GetNdSbpValidTransformationAxisSequence(Symbol<NdSbp> src_nd_sbp,
                                                                    Symbol<NdSbp> dst_nd_sbp) {
  HashMap<int64_t, int64_t> dst_nd_sbp_axis2exclusive_src_nd_sbp_axis;
  std::function<Maybe<Optional<int64_t>>(int64_t)> ExclusiveSrcNdSbpAxis4DstNdSbpAxis;
  JUST(MakeExclusiveSrcNdSbpAxis4DstNdSbpAxis(&ExclusiveSrcNdSbpAxis4DstNdSbpAxis, src_nd_sbp,
                                              dst_nd_sbp));
  bool is_acyclic = JUST(
      IsNdSbpBoxingAcyclic(src_nd_sbp->sbp_parallel_size(), ExclusiveSrcNdSbpAxis4DstNdSbpAxis));
  CHECK_OR_RETURN(is_acyclic) << Error::UnimplementedError()
                              << "cyclic split axis boxing are not supported";
  std::vector<int64_t> nd_sbp_axis_sequence;
  JUST(InitNdSbpValidTransformationAxisSequence(&nd_sbp_axis_sequence, src_nd_sbp, dst_nd_sbp,
                                                ExclusiveSrcNdSbpAxis4DstNdSbpAxis));
  return nd_sbp_axis_sequence;
}

std::string GetCyclicBoxingDebugString(
    Symbol<NdSbp> src_nd_sbp, Symbol<NdSbp> dst_nd_sbp,
    const std::function<Maybe<Optional<int64_t>>(int64_t)>& ExclusiveSrcNdSbpAxis4DstNdSbpAxis) {
  CHECK_EQ(src_nd_sbp->sbp_parallel_size(), dst_nd_sbp->sbp_parallel_size());
  std::stringstream ss;
  ss << "cyclic split axis boxing are not supported. "
     << "src_nd_sbp: " << NdSbpToString(src_nd_sbp) << ", dst_nd_sbp: " << NdSbpToString(dst_nd_sbp)
     << ". "
     << "dst_nd_sbp axis to exclusive src_nd_sbp axis: ";
  ss << "[";
  for (int i = 0; i < src_nd_sbp->sbp_parallel_size(); ++i) {
    const auto& opt_axis = CHECK_JUST(ExclusiveSrcNdSbpAxis4DstNdSbpAxis(i));
    if (i) { ss << ", "; }
    if (opt_axis->has_value()) {
      ss << CHECK_JUST(*opt_axis);
    } else {
      ss << "None";
    }
  }
  ss << "]";
  return ss.str();
}

Maybe<Shape> GetPhysicalShape(const Shape& shape, Symbol<NdSbp> nd_sbp,
                              Symbol<ParallelDesc> parallel_desc) {
  const auto& parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
  return GetPhysicalShape(shape, *nd_sbp, *parallel_desc, JUST(*parallel_id));
}

Maybe<Shape> GetSubLogicalShape(Symbol<one::GlobalTensorMeta> tensor_meta,
                                Symbol<ParallelDesc> sub_parallel_desc, Symbol<NdSbp> sub_nd_sbp) {
  CHECK_EQ_OR_RETURN(sub_nd_sbp->sbp_parallel_size(), 1);  // NOLINT(maybe-need-error-msg)
  const auto& logical_shape = tensor_meta->shape();
  const auto& physical_shape =
      JUST(GetPhysicalShape(logical_shape, tensor_meta->nd_sbp(), tensor_meta->parallel_desc()));

  std::shared_ptr<Shape> sub_logical_shape = std::make_shared<Shape>(*physical_shape);
  if (sub_nd_sbp->sbp_parallel(0).has_split_parallel()) {
    const int64_t split_axis = sub_nd_sbp->sbp_parallel(0).split_parallel().axis();
    sub_logical_shape->Set(split_axis, logical_shape.At(split_axis));
  }
  return sub_logical_shape;
}

Maybe<Symbol<one::GlobalTensorMeta>> CalcSubGlobalTensorMeta(
    Symbol<one::GlobalTensorMeta> tensor_meta, Symbol<ParallelDesc> sub_parallel_desc,
    Symbol<NdSbp> sub_nd_sbp) {
  CHECK_EQ_OR_RETURN(sub_nd_sbp->sbp_parallel_size(), 1);  // NOLINT(maybe-need-error-msg)
  const auto& logical_shape = JUST(GetSubLogicalShape(tensor_meta, sub_parallel_desc, sub_nd_sbp));
  one::GlobalTensorMeta sub_global_tensor_meta(*logical_shape, tensor_meta->dtype(), sub_nd_sbp,
                                               sub_parallel_desc);
  return SymbolOf(sub_global_tensor_meta);
}

static constexpr auto* GetSubGlobalTensorMeta = DECORATE(&CalcSubGlobalTensorMeta, ThreadLocal);

Maybe<Symbol<NdSbp>> ReplaceNdSbpComponent(Symbol<NdSbp> nd_sbp, int64_t axis,
                                           Symbol<NdSbp> component) {
  CHECK_GE_OR_RETURN(axis, 0);
  CHECK_LT_OR_RETURN(axis, nd_sbp->sbp_parallel_size());
  CHECK_EQ_OR_RETURN(component->sbp_parallel_size(), 1);
  NdSbp new_nd_sbp(*nd_sbp);
  *new_nd_sbp.mutable_sbp_parallel(axis) = component->sbp_parallel(0);
  return SymbolOf(new_nd_sbp);
}

Maybe<Symbol<one::GlobalTensorMeta>> ReplaceNdSbp(Symbol<one::GlobalTensorMeta> tensor_meta,
                                                  Symbol<NdSbp> nd_sbp) {
  one::GlobalTensorMeta new_tensor_meta(tensor_meta->shape(), tensor_meta->dtype(), nd_sbp,
                                        tensor_meta->parallel_desc());
  return SymbolOf(new_tensor_meta);
}

Maybe<std::vector<NaiveBoxingTransformation>> DecomposeIntoNaiveTransformations(
    Symbol<one::GlobalTensorMeta> tensor_meta, Symbol<NdSbp> dst_nd_sbp) {
  std::tie(tensor_meta, dst_nd_sbp) = *JUST(GetDecomposableEquivalent(tensor_meta, dst_nd_sbp));
  const auto& parallel_desc = tensor_meta->parallel_desc();
  const auto& src_nd_sbp = tensor_meta->nd_sbp();
  CHECK_EQ_OR_RETURN(src_nd_sbp->sbp_parallel_size(), dst_nd_sbp->sbp_parallel_size());
  std::vector<int64_t> nd_sbp_axis_sequence;
  {
    std::function<Maybe<Optional<int64_t>>(int64_t)> ExclusiveSrcNdSbpAxis4DstNdSbpAxis;
    JUST(MakeExclusiveSrcNdSbpAxis4DstNdSbpAxis(&ExclusiveSrcNdSbpAxis4DstNdSbpAxis, src_nd_sbp,
                                                dst_nd_sbp));
    bool is_acyclic = JUST(
        IsNdSbpBoxingAcyclic(src_nd_sbp->sbp_parallel_size(), ExclusiveSrcNdSbpAxis4DstNdSbpAxis));
    CHECK_OR_RETURN(is_acyclic) << Error::UnimplementedError()
                                << GetCyclicBoxingDebugString(src_nd_sbp, dst_nd_sbp,
                                                              ExclusiveSrcNdSbpAxis4DstNdSbpAxis);
    JUST(InitNdSbpValidTransformationAxisSequence(&nd_sbp_axis_sequence, src_nd_sbp, dst_nd_sbp,
                                                  ExclusiveSrcNdSbpAxis4DstNdSbpAxis));
  }
  const auto& transformations = std::make_shared<std::vector<NaiveBoxingTransformation>>();
  for (int axis : nd_sbp_axis_sequence) {
    const auto& src_sbp = src_nd_sbp->sbp_parallel(axis);
    const auto& dst_sbp = dst_nd_sbp->sbp_parallel(axis);
    if (src_sbp == dst_sbp) { continue; }
    std::vector<int> axis2selected(src_nd_sbp->sbp_parallel_size());
    axis2selected[axis] = 1;
    const auto& sub_parallel_desc =
        JUST(GetSelectedSubParallelDesc(parallel_desc, SymbolOf(axis2selected)));
    const auto& sub_src_nd_sbp = JUST(MakeNdSbp(src_sbp));
    const auto& sub_dst_nd_sbp = JUST(MakeNdSbp(dst_sbp));
    const auto& sub_global_tensor_meta =
        JUST(GetSubGlobalTensorMeta(tensor_meta, sub_parallel_desc, sub_src_nd_sbp));
    const auto& new_src_nd_sbp =
        JUST(ReplaceNdSbpComponent(tensor_meta->nd_sbp(), axis, sub_dst_nd_sbp));
    tensor_meta = JUST(ReplaceNdSbp(tensor_meta, new_src_nd_sbp));
    transformations->emplace_back(NaiveBoxingTransformation{
        .global_tensor_meta = sub_global_tensor_meta,
        .dst_nd_sbp = sub_dst_nd_sbp,
    });
  }
  return transformations;
}

}  // namespace private_details

namespace {

Maybe<std::unordered_map<int64_t, Symbol<ParallelDesc>>> CalcBroadcastGroup(
    Symbol<ParallelDesc> src_parallel_desc, Symbol<ParallelDesc> dst_parallel_desc,
    bool allow_across_node) {
  CHECK_EQ_OR_RETURN(src_parallel_desc->parallel_num(),
                     src_parallel_desc->sorted_machine_ids().size());
  CHECK_EQ_OR_RETURN(dst_parallel_desc->parallel_num(),
                     dst_parallel_desc->sorted_machine_ids().size());
  CHECK_EQ_OR_RETURN(src_parallel_desc->device_type(), dst_parallel_desc->device_type());
  CHECK_LE_OR_RETURN(src_parallel_desc->parallel_num(), dst_parallel_desc->parallel_num());
  const auto& src_process_ids = src_parallel_desc->sorted_machine_ids();
  HashMap<int64_t, std::vector<int64_t>> process_id2group{};
  HashMap<int64_t, std::vector<int64_t>> node_id2src_process_id{};
  for (int64_t process_id : src_process_ids) {
    std::vector<int64_t> vec{process_id};
    CHECK_OR_RETURN(process_id2group.emplace(process_id, vec).second);
    CHECK_OR_RETURN(dst_parallel_desc->ContainingMachineId(process_id));
    node_id2src_process_id[GlobalProcessCtx::NodeId(process_id)].emplace_back(process_id);
  }
  std::vector<int64_t> remainder_process_ids{};
  remainder_process_ids.reserve(dst_parallel_desc->sorted_machine_ids().size());
  HashMap<int64_t, int64_t> node_id2counter{};
  for (int64_t process_id : dst_parallel_desc->sorted_machine_ids()) {
    if (!src_parallel_desc->ContainingMachineId(process_id)) {
      const auto& node_iter = node_id2src_process_id.find(GlobalProcessCtx::NodeId(process_id));
      if (node_iter == node_id2src_process_id.end()) {
        CHECK_OR_RETURN(allow_across_node)
            << Error::UnimplementedError() << "\n----[src_placement]----\n"
            << src_parallel_desc->parallel_conf().DebugString() << "\n----[dst_placement]----\n"
            << dst_parallel_desc->parallel_conf().DebugString();
        // handle `process_id` later.
        remainder_process_ids.emplace_back(process_id);
      } else {
        // balancedly put `process_id` into the groups within the same node..
        int64_t node_id = node_iter->first;
        const auto& src_process_ids = node_iter->second;
        int64_t src_process_index = (node_id2counter[node_id]++) % src_process_ids.size();
        int64_t src_process_id = src_process_ids.at(src_process_index);
        JUST(MapAt(process_id2group, src_process_id)).emplace_back(process_id);
      }
    }
  }
  // put remainder process ids into src groups.
  for (int i = 0; i < remainder_process_ids.size(); ++i) {
    int64_t src_process_id = src_process_ids.at(i % src_process_ids.size());
    JUST(MapAt(process_id2group, src_process_id))
        .emplace_back(JUST(oneflow::VectorAt(remainder_process_ids, i)));
  }
  const auto& map = std::make_shared<std::unordered_map<int64_t, Symbol<ParallelDesc>>>();
  for (const auto& pair : process_id2group) {
    const auto& group = pair.second;
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag(dst_parallel_desc->parallel_conf().device_tag());
    for (int64_t process_id : group) {
      const auto& device_ids = dst_parallel_desc->sorted_dev_phy_ids(process_id);
      CHECK_EQ_OR_RETURN(device_ids.size(), 1);
      parallel_conf.add_device_name(std::string("@") + std::to_string(process_id) + ":"
                                    + std::to_string(device_ids.at(0)));
    }
    const auto& parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
    for (int64_t process_id : group) {
      CHECK_OR_RETURN(map->emplace(process_id, parallel_desc).second);
    }
  }
  return map;
}
auto* CachedBroadcastGroup = DECORATE(&CalcBroadcastGroup, ThreadLocal);

Maybe<void> RawCheckIsNdSbpBoxingAcyclic(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  using namespace private_details;
  const auto& src_nd_sbp = in->nd_sbp();
  const auto& dst_nd_sbp = out->nd_sbp();
  std::function<Maybe<Optional<int64_t>>(int64_t)> ExclusiveSrcNdSbpAxis4DstNdSbpAxis;
  JUST(MakeExclusiveSrcNdSbpAxis4DstNdSbpAxis(&ExclusiveSrcNdSbpAxis4DstNdSbpAxis, src_nd_sbp,
                                              dst_nd_sbp));
  bool is_acyclic = JUST(
      IsNdSbpBoxingAcyclic(src_nd_sbp->sbp_parallel_size(), ExclusiveSrcNdSbpAxis4DstNdSbpAxis));
  CHECK_OR_RETURN(is_acyclic) << Error::UnimplementedError()
                              << GetCyclicBoxingDebugString(src_nd_sbp, dst_nd_sbp,
                                                            ExclusiveSrcNdSbpAxis4DstNdSbpAxis);
  return Maybe<void>::Ok();
}

Maybe<void> RawCheckIsNdSbpBoxingAcyclicWithDecompose(Symbol<PlacedNdSbp> in,
                                                      Symbol<PlacedNdSbp> out,
                                                      const Shape& logical_shape) {
  using namespace private_details;
  Symbol<NdSbp> src_nd_sbp = in->nd_sbp();
  Symbol<NdSbp> dst_nd_sbp = out->nd_sbp();
  const auto& hierarchy = in->placement()->hierarchy();
  std::shared_ptr<const Shape> shape;

  std::tie(shape, src_nd_sbp, dst_nd_sbp) = *JUST(CalcDecomposableEquivalentShapeAndNdSbpPair(
      logical_shape, *hierarchy, src_nd_sbp, dst_nd_sbp));

  std::function<Maybe<Optional<int64_t>>(int64_t)> ExclusiveSrcNdSbpAxis4DstNdSbpAxis;
  JUST(MakeExclusiveSrcNdSbpAxis4DstNdSbpAxis(&ExclusiveSrcNdSbpAxis4DstNdSbpAxis, src_nd_sbp,
                                              dst_nd_sbp));
  bool is_acyclic = JUST(
      IsNdSbpBoxingAcyclic(src_nd_sbp->sbp_parallel_size(), ExclusiveSrcNdSbpAxis4DstNdSbpAxis));
  CHECK_OR_RETURN(is_acyclic) << Error::UnimplementedError()
                              << GetCyclicBoxingDebugString(src_nd_sbp, dst_nd_sbp,
                                                            ExclusiveSrcNdSbpAxis4DstNdSbpAxis);
  return Maybe<void>::Ok();
}

}  // namespace

int64_t CalcIndex4Axis(int64_t offset, const Stride& stride, int axis) {
  CHECK_LT(axis, stride.size()) << "Expected axis (" << axis << ") to be less than size of stride ("
                                << stride.size() << ")";
  if (axis == 0) {
    return offset / stride.at(0);
  } else {
    return offset % stride.at(axis - 1) / stride.at(axis);
  }
}

decltype(CheckIsNdSbpBoxingAcyclic) CheckIsNdSbpBoxingAcyclic =
    DECORATE(&RawCheckIsNdSbpBoxingAcyclic, ThreadLocal);

decltype(CheckIsNdSbpBoxingAcyclicWithDecompose) CheckIsNdSbpBoxingAcyclicWithDecompose =
    DECORATE(&RawCheckIsNdSbpBoxingAcyclicWithDecompose, ThreadLocalCopiable);

Maybe<std::unordered_map<int64_t, Symbol<ParallelDesc>>> GetBroadcastGroup(
    Symbol<ParallelDesc> src_parallel_desc, Symbol<ParallelDesc> dst_parallel_desc) {
  return CachedBroadcastGroup(src_parallel_desc, dst_parallel_desc, true);
}

Maybe<std::unordered_map<int64_t, Symbol<ParallelDesc>>> GetBroadcastGroupWithoutAcrossNode(
    Symbol<ParallelDesc> src_parallel_desc, Symbol<ParallelDesc> dst_parallel_desc) {
  return CachedBroadcastGroup(src_parallel_desc, dst_parallel_desc, false);
}

}  // namespace oneflow
