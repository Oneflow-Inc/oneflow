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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

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

Maybe<Symbol<ParallelDesc>> GetBroadcastSubParallelDesc(const ParallelDesc& parallel_desc,
                                                        const cfg::NdSbp& nd_sbp,
                                                        int64_t parallel_id) {
  const auto& hierarchy_shape = *parallel_desc.hierarchy();
  std::vector<bool> dim2is_broadcast(nd_sbp.sbp_parallel_size());
  for (int i = 0; i < dim2is_broadcast.size(); ++i) {
    dim2is_broadcast.at(i) = nd_sbp.sbp_parallel(i).has_broadcast_parallel();
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

Maybe<Symbol<ParallelDesc>> GetBroadcastSubParallelDesc(Symbol<ParallelDesc> parallel_desc,
                                                        Symbol<cfg::NdSbp> nd_sbp) {
  using PlacementSbp = std::pair<Symbol<ParallelDesc>, Symbol<cfg::NdSbp>>;
  static thread_local HashMap<PlacementSbp, Symbol<ParallelDesc>> map;
  const auto& key = std::make_pair(parallel_desc, nd_sbp);
  auto iter = map.find(key);
  if (iter == map.end()) {
    Optional<int64_t> opt_parallel_id;
    JUST(GetDevice4CurrentProcessCtx(parallel_desc, &opt_parallel_id));
    int64_t parallel_id = JUST(opt_parallel_id.value());
    const auto& sub_parallel_desc =
        JUST(GetBroadcastSubParallelDesc(*parallel_desc, *nd_sbp, parallel_id));
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
    node_id2src_process_id[GlobalProcessCtx::NodeId(process_id)].push_back(process_id);
  }
  std::vector<int64_t> remainder_process_ids{};
  remainder_process_ids.reserve(dst_parallel_desc->sorted_machine_ids().size());
  HashMap<int64_t, int64_t> node_id2counter{};
  for (int64_t process_id : dst_parallel_desc->sorted_machine_ids()) {
    if (!src_parallel_desc->ContainingMachineId(process_id)) {
      const auto& node_iter = node_id2src_process_id.find(GlobalProcessCtx::NodeId(process_id));
      if (node_iter == node_id2src_process_id.end()) {
        CHECK_OR_RETURN(allow_across_node)
            << Error::Unimplemented() << "\n----[src_placement]----\n"
            << src_parallel_desc->parallel_conf().DebugString() << "\n----[dst_placement]----\n"
            << dst_parallel_desc->parallel_conf().DebugString();
        // handle `process_id` later.
        remainder_process_ids.push_back(process_id);
      } else {
        // balancedly put `process_id` into the groups within the same node..
        int64_t node_id = node_iter->first;
        const auto& src_process_ids = node_iter->second;
        int64_t src_process_index = (node_id2counter[node_id]++) % src_process_ids.size();
        int64_t src_process_id = src_process_ids.at(src_process_index);
        JUST(MutMapAt(&process_id2group, src_process_id))->push_back(process_id);
      }
    }
  }
  // put remainder process ids into src groups.
  for (int i = 0; i < remainder_process_ids.size(); ++i) {
    int64_t src_process_id = src_process_ids.at(i % src_process_ids.size());
    JUST(MutMapAt(&process_id2group, src_process_id))->push_back(remainder_process_ids.at(i));
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

}  // namespace

Maybe<std::unordered_map<int64_t, Symbol<ParallelDesc>>> GetBroadcastGroup(
    Symbol<ParallelDesc> src_parallel_desc, Symbol<ParallelDesc> dst_parallel_desc) {
  return CachedBroadcastGroup(src_parallel_desc, dst_parallel_desc, true);
}

Maybe<std::unordered_map<int64_t, Symbol<ParallelDesc>>> GetBroadcastGroupWithoutAcrossNode(
    Symbol<ParallelDesc> src_parallel_desc, Symbol<ParallelDesc> dst_parallel_desc) {
  return CachedBroadcastGroup(src_parallel_desc, dst_parallel_desc, false);
}

}  // namespace oneflow
