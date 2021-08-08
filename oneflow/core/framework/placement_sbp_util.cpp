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
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace oneflow {

namespace private_details {

namespace {

using IndexVector = DimVector;
using StrideVector = DimVector;

void GetStrideVector(const Shape& shape, StrideVector* strides) {
  strides->resize(shape.NumAxes());
  for (int i = 0; i < shape.NumAxes(); ++i) { strides->at(i) = shape.Count(i + 1); }
}

Maybe<void> GetIndexesFromOffset(const StrideVector& strides, int64_t offset,
                                 IndexVector* indexes) {
  indexes->resize(strides.size());
  for (int i = 0; i < strides.size(); ++i) {
    indexes->at(i) = offset / strides.at(i);
    offset = offset % strides.at(i);
  }
  CHECK_EQ_OR_RETURN(offset, 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetOffsetFromIndexes(const StrideVector& strides, const IndexVector& indexes,
                                 int64_t* offset) {
  CHECK_EQ_OR_RETURN(strides.size(), indexes.size());
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

Maybe<Symbol<std::vector<int>>> CalcAxis2IsBroadcast(
    Symbol<cfg::ParallelDistribution> parallel_distribution) {
  std::vector<int> axis2is_selected(parallel_distribution->sbp_parallel_size());
  for (int i = 0; i < axis2is_selected.size(); ++i) {
    axis2is_selected.at(i) = parallel_distribution->sbp_parallel(i).has_broadcast_parallel();
  }
  return SymbolOf(axis2is_selected);
}

static auto* GetAxis2IsBroadcast = DECORATE(&CalcAxis2IsBroadcast, ThreadLocal);

Maybe<Symbol<ParallelDesc>> CalcSelectedSubParallelDesc(Symbol<ParallelDesc> parallel_desc,
                                                        Symbol<std::vector<int>> axis2is_selected,
                                                        int64_t parallel_id) {
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

Maybe<std::vector<int64_t>> GetSelectedParallelIds(const Shape& hierarchy_shape,
                                                   const std::vector<int>& axis2is_selected,
                                                   int64_t parallel_id) {
  CHECK_EQ_OR_RETURN(hierarchy_shape.NumAxes(), axis2is_selected.size());
  StrideVector hierarchy_strides{};
  GetStrideVector(hierarchy_shape, &hierarchy_strides);
  IndexVector indexes{};
  JUST(GetIndexesFromOffset(hierarchy_strides, parallel_id, &indexes));
  std::function<void(const DimVector&, DimVector*)> SelectedIndex2OriginIndex;
  JUST(GetSelectedIndex2OriginIndex(indexes, axis2is_selected, &SelectedIndex2OriginIndex));
  const auto& broadcast_shape = JUST(GetSelectedShape(hierarchy_shape, axis2is_selected));
  StrideVector broadcast_strides{};
  GetStrideVector(*broadcast_shape, &broadcast_strides);
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

Maybe<Symbol<ParallelDesc>> GetBroadcastSubParallelDesc(
    Symbol<ParallelDesc> parallel_desc, Symbol<cfg::ParallelDistribution> parallel_distribution) {
  Optional<int64_t> opt_parallel_id;
  JUST(GetDevice4CurrentProcessCtx(parallel_desc, &opt_parallel_id));
  int64_t parallel_id = JUST(opt_parallel_id.value());
  const auto& axis2is_selected = JUST(GetAxis2IsBroadcast(parallel_distribution));
  return GetSelectedSubParallelDesc(parallel_desc, axis2is_selected, parallel_id);
}

namespace {

Maybe<Symbol<cfg::ParallelDistribution>> MakeNdSbp(const cfg::SbpParallel& sbp) {
  cfg::ParallelDistribution nd_sbp;
  nd_sbp.mutable_sbp_parallel()->Add()->CopyFrom(sbp);
  return SymbolOf(nd_sbp);
}

}  // namespace

Maybe<std::vector<NaiveBoxingTransformation>> DecomposeByParallelId(
    Symbol<ParallelDesc> parallel_desc, int64_t parallel_id,
    Symbol<cfg::ParallelDistribution> src_nd_sbp, Symbol<cfg::ParallelDistribution> dst_nd_sbp) {
  CHECK_EQ_OR_RETURN(src_nd_sbp->sbp_parallel_size(), dst_nd_sbp->sbp_parallel_size());
  const auto& transformations = std::make_shared<std::vector<NaiveBoxingTransformation>>();
  for (int i = 0; i < src_nd_sbp->sbp_parallel_size(); ++i) {
    const auto& src_sbp = src_nd_sbp->sbp_parallel(i);
    const auto& dst_sbp = dst_nd_sbp->sbp_parallel(i);
    if (src_sbp == dst_sbp) { continue; }
    std::vector<int> axis2selected(src_nd_sbp->sbp_parallel_size());
    axis2selected[i] = 1;
    const auto& sub_parallel_desc =
        JUST(GetSelectedSubParallelDesc(parallel_desc, SymbolOf(axis2selected), parallel_id));
    transformations->push_back(NaiveBoxingTransformation{
        .parallel_desc = sub_parallel_desc,
        .src_nd_sbp = JUST(MakeNdSbp(src_sbp)),
        .dst_nd_sbp = JUST(MakeNdSbp(dst_sbp)),
    });
  }
  return transformations;
}

}  // namespace private_details

static auto* DecomposeByParallelId = DECORATE(&private_details::DecomposeByParallelId, ThreadLocal);

Maybe<std::vector<NaiveBoxingTransformation>> DecomposeIntoNaiveTransformations(
    Symbol<ParallelDesc> parallel_desc, Symbol<cfg::ParallelDistribution> src_nd_sbp,
    Symbol<cfg::ParallelDistribution> dst_nd_sbp) {
  Optional<int64_t> opt_parallel_id;
  JUST(GetDevice4CurrentProcessCtx(parallel_desc, &opt_parallel_id));
  int64_t parallel_id = JUST(opt_parallel_id.value());
  return DecomposeByParallelId(parallel_desc, parallel_id, src_nd_sbp, dst_nd_sbp);
}

}  // namespace oneflow
