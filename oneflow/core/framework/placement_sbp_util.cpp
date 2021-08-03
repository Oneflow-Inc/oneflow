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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace oneflow {

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

Maybe<void> GetBroadcastIndex2OriginIndex(
    const IndexVector& indexes, const std::vector<bool>& dim2is_broadcast,
    std::function<void(const DimVector&, DimVector*)>* BroadcastIndex2OriginIndex) {
  CHECK_EQ_OR_RETURN(dim2is_broadcast.size(), indexes.size());
  *BroadcastIndex2OriginIndex = [=](const DimVector& broadcast, DimVector* origin) {
    origin->resize(indexes.size());
    for (int i = 0; i < indexes.size(); ++i) {
      origin->at(i) = dim2is_broadcast.at(i) ? broadcast.at(i) : indexes.at(i);
    }
  };
  return Maybe<void>::Ok();
}

Maybe<const Shape> GetBroadcastShape(const Shape& hierarchy_shape,
                                     const std::vector<bool>& dim2is_broadcast) {
  CHECK_EQ_OR_RETURN(hierarchy_shape.NumAxes(), dim2is_broadcast.size());
  DimVector dim_vec = hierarchy_shape.dim_vec();
  for (int i = 0; i < dim2is_broadcast.size(); ++i) {
    if (!dim2is_broadcast.at(i)) { dim_vec.at(i) = 1; }
  }
  return std::make_shared<const Shape>(dim_vec);
}

Maybe<Symbol<ParallelDesc>> GetBroadcastSubParallelDesc(
    const ParallelDesc& parallel_desc, const cfg::ParallelDistribution& parallel_distribution,
    int64_t parallel_id) {
  const auto& hierarchy_shape = *parallel_desc.hierarchy();
  std::vector<bool> dim2is_broadcast(parallel_distribution.sbp_parallel_size());
  for (int i = 0; i < dim2is_broadcast.size(); ++i) {
    dim2is_broadcast.at(i) = parallel_distribution.sbp_parallel(i).has_broadcast_parallel();
  }
  const auto& broadcast_parallel_ids =
      JUST(GetBroadcastParallelIds(hierarchy_shape, dim2is_broadcast, parallel_id));
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(parallel_desc.device_tag());
  bool found_parallel_id = false;
  for (int64_t i : *broadcast_parallel_ids) {
    found_parallel_id = found_parallel_id || (i == parallel_id);
    int64_t machine_id = JUST(parallel_desc.MachineId4ParallelId(i));
    int64_t device_id = JUST(parallel_desc.DeviceId4ParallelId(i));
    parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":"
                                  + std::to_string(device_id));
  }
  CHECK_OR_RETURN(found_parallel_id);
  return SymbolOf(ParallelDesc(parallel_conf));
}

}  // namespace

Maybe<Symbol<ParallelDesc>> GetBroadcastSubParallelDesc(
    Symbol<ParallelDesc> parallel_desc, Symbol<cfg::ParallelDistribution> parallel_distribution) {
  using PlacementSbp = std::pair<Symbol<ParallelDesc>, Symbol<cfg::ParallelDistribution>>;
  static thread_local HashMap<PlacementSbp, Symbol<ParallelDesc>> map;
  const auto& key = std::make_pair(parallel_desc, parallel_distribution);
  auto iter = map.find(key);
  if (iter == map.end()) {
    Optional<int64_t> opt_parallel_id;
    JUST(GetDevice4CurrentProcessCtx(parallel_desc, &opt_parallel_id));
    int64_t parallel_id = JUST(opt_parallel_id.value());
    const auto& sub_parallel_desc =
        JUST(GetBroadcastSubParallelDesc(*parallel_desc, *parallel_distribution, parallel_id));
    iter = map.emplace(key, sub_parallel_desc).first;
  }
  return iter->second;
}

Maybe<std::vector<int64_t>> GetBroadcastParallelIds(const Shape& hierarchy_shape,
                                                    const std::vector<bool>& dim2is_broadcast,
                                                    int64_t parallel_id) {
  CHECK_EQ_OR_RETURN(hierarchy_shape.NumAxes(), dim2is_broadcast.size());
  StrideVector hierarchy_strides{};
  GetStrideVector(hierarchy_shape, &hierarchy_strides);
  IndexVector indexes{};
  JUST(GetIndexesFromOffset(hierarchy_strides, parallel_id, &indexes));
  std::function<void(const DimVector&, DimVector*)> BroadcastIndex2OriginIndex;
  JUST(GetBroadcastIndex2OriginIndex(indexes, dim2is_broadcast, &BroadcastIndex2OriginIndex));
  const auto& broadcast_shape = JUST(GetBroadcastShape(hierarchy_shape, dim2is_broadcast));
  StrideVector broadcast_strides{};
  GetStrideVector(*broadcast_shape, &broadcast_strides);
  const auto& origin_offsets = std::make_shared<std::vector<int64_t>>(broadcast_shape->elem_cnt());
  for (int64_t i = 0; i < broadcast_shape->elem_cnt(); ++i) {
    IndexVector broadcast_indexes{};
    JUST(GetIndexesFromOffset(broadcast_strides, i, &broadcast_indexes));
    IndexVector origin_indexes{};
    BroadcastIndex2OriginIndex(broadcast_indexes, &origin_indexes);
    int64_t origin_offset = -1;
    JUST(GetOffsetFromIndexes(hierarchy_strides, origin_indexes, &origin_offset));
    origin_offsets->at(i) = origin_offset;
  }
  return origin_offsets;
}

}  // namespace oneflow
