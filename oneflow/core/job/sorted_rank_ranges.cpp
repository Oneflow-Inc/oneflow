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
#include "oneflow/core/job/sorted_rank_ranges.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

/*static*/ Maybe<Symbol<SortedRankRanges>> SortedRankRanges::New4SoleDevicePerRankParallelDesc(
    Symbol<ParallelDesc> parallel_desc) {
  static thread_local HashMap<Symbol<ParallelDesc>, Symbol<SortedRankRanges>> map;
  auto iter = map.find(parallel_desc);
  if (iter == map.end()) {
    const auto& sorted_rank_ranges = JUST(New4SoleDevicePerRankParallelDesc(*parallel_desc));
    iter = map.emplace(parallel_desc, SymbolOf(*sorted_rank_ranges)).first;
  }
  return iter->second;
}

/*static*/ Maybe<SortedRankRanges> SortedRankRanges::New4SoleDevicePerRankParallelDesc(
    const ParallelDesc& parallel_desc) {
  CHECK_GT_OR_RETURN(parallel_desc.sorted_machine_ids().size(), 0);
  CHECK_EQ_OR_RETURN(parallel_desc.sorted_machine_ids().size(), parallel_desc.parallel_num());
  std::shared_ptr<SortedRankRanges> sorted_rank_ranges(new SortedRankRanges());
  // Initialize sorted_rank_ranges_
  {
    std::vector<Range>* machine_id_ranges = &sorted_rank_ranges.get()->sorted_rank_ranges_;
    for (int64_t machine_id : parallel_desc.sorted_machine_ids()) {
      if (!machine_id_ranges->empty() && machine_id_ranges->back().end() + 1 == machine_id) {
        // Range* last = const_cast<Range*>(&machine_id_ranges->at(machine_id_ranges->size() - 1));
        Range* last = &machine_id_ranges->at(machine_id_ranges->size() - 1);
        last->mut_end() = machine_id;
      } else {
        machine_id_ranges->push_back(Range(machine_id, machine_id));
      }
    }
  }
	// Initialize rank2next_rank_in_ring_
	{
		const auto& ranges = sorted_rank_ranges.get()->sorted_rank_ranges_;
		int64_t last = ranges.at(ranges.size() - 1).end();
    for (const auto& machine_id_range : ranges) {
			for (int64_t i = machine_id_range.begin(); i <= machine_id_range.end(); ++i) {
				CHECK_OR_RETURN(sorted_rank_ranges.get()->rank2next_rank_in_ring_.emplace(last, i).second);
				last = i;
			}
		}
	}
  // Initialize rpc_push_pull_key_
  {
    int i = 0;
    std::stringstream ss;
    for (const auto& machine_id_range : sorted_rank_ranges.get()->sorted_rank_ranges_) {
      if (i++ > 0) { ss << ","; }
      ss << machine_id_range.begin() << "-" << machine_id_range.end();
    }
    sorted_rank_ranges.get()->rpc_push_pull_key_ = ss.str();
  }
  // Initialize hash_value_
  {
    sorted_rank_ranges.get()->hash_value_ = 0;
    const auto& hash_functor = std::hash<Range>();
    for (const auto& range : sorted_rank_ranges->sorted_rank_ranges_) {
      HashCombine(&sorted_rank_ranges->hash_value_, hash_functor(range));
    }
  }
  return sorted_rank_ranges;
}

Maybe<int64_t> SortedRankRanges::GetNextRankInRing(int64_t rank) const {
	return MapAt(rank2next_rank_in_ring_, rank);
}

}  // namespace oneflow
