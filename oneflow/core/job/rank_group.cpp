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
#include <map>
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

/*static*/ Maybe<Symbol<RankGroup>> RankGroup::New(Symbol<ParallelDesc> parallel_desc) {
  return DECORATE(&RankGroup::RawNew, ThreadLocal)(parallel_desc);
}

/*static*/ Maybe<Symbol<RankGroup>> RankGroup::RawNew(Symbol<ParallelDesc> parallel_desc) {
  CHECK_EQ_OR_RETURN(parallel_desc->sorted_machine_ids().size(), parallel_desc->parallel_num());
  const auto& sorted_machine_ids = parallel_desc->sorted_machine_ids();
  return New(std::set<int64_t>{sorted_machine_ids.begin(), sorted_machine_ids.end()});
}

/*static*/ Maybe<Symbol<RankGroup>> RankGroup::New(const std::set<int64_t>& ranks) {
  static thread_local std::map<std::set<int64_t>, Symbol<RankGroup>> map;
  auto iter = map.find(ranks);
  if (iter == map.end()) {
    RankGroup rank_group;
    JUST(rank_group.Init(ranks));
    iter = map.emplace(ranks, SymbolOf(rank_group)).first;
  }
  return iter->second;
}

namespace {

Maybe<Symbol<ParallelDesc>> CalcDefaultParallelDesc(DeviceType device_type,
                                                    Symbol<RankGroup> rank_group) {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(*JUST(DeviceTag4DeviceType(device_type)));
  JUST(rank_group->ForEachRank([&](int64_t rank) -> Maybe<void> {
    int64_t local_rank = GlobalProcessCtx::LocalRank(rank);
    parallel_conf.add_device_name(std::string("@") + std::to_string(rank) + ":"
                                  + std::to_string(local_rank));
    return Maybe<void>::Ok();
  }));
  return SymbolOf(ParallelDesc(parallel_conf));
}

auto* CachedDefaultParallelDesc = DECORATE(&CalcDefaultParallelDesc, ThreadLocal);

}  // namespace

/*static*/ Maybe<Symbol<ParallelDesc>> RankGroup::GetDefaultParallelDesc(
    DeviceType device_type, Symbol<RankGroup> rank_group) {
  return CachedDefaultParallelDesc(device_type, rank_group);
}

namespace {

Maybe<std::set<int64_t>> AllWorldRanks() {
  const auto& ranks = std::make_shared<std::set<int64_t>>();
  for (int i = 0; i < GlobalProcessCtx::WorldSize(); ++i) { ranks->insert(i); }
  return ranks;
}

}  // namespace

/*static*/ Maybe<Symbol<RankGroup>> RankGroup::DefaultRankGroup() {
  const auto& all_wold_ranks = JUST(AllWorldRanks());
  const auto& rank_group = JUST(RankGroup::New(*all_wold_ranks));
  return rank_group;
}

Maybe<void> RankGroup::Init(const std::set<int64_t>& ranks) {
  ranks_ = ranks;
  // Initialize rank2next_rank_in_ring_ and rank2prev_rank_in_ring_
  {
    CHECK_GT_OR_RETURN(ranks.size(), 0);
    int64_t last = *(--ranks.end());
    for (int64_t i : ranks) {
      CHECK_OR_RETURN(rank2next_rank_in_ring_.emplace(last, i).second);
      CHECK_OR_RETURN(rank2prev_rank_in_ring_.emplace(i, last).second);
      last = i;
    }
  }
  // Initialize hash_value_
  hash_value_ = 0;
  for (int64_t i : ranks) { HashCombine(&hash_value_, i); }
  return Maybe<void>::Ok();
}

Maybe<int64_t> RankGroup::GetNextRankInRing(int64_t rank) const {
  return MapAt(rank2next_rank_in_ring_, rank);
}

Maybe<int64_t> RankGroup::GetNextRankInRing() const {
  return GetNextRankInRing(GlobalProcessCtx::Rank());
}

Maybe<int64_t> RankGroup::GetPrevRankInRing(int64_t rank) const {
  return MapAt(rank2prev_rank_in_ring_, rank);
}

Maybe<int64_t> RankGroup::GetPrevRankInRing() const {
  return GetPrevRankInRing(GlobalProcessCtx::Rank());
}

bool RankGroup::ContainingCurrentRank() const {
  return rank2next_rank_in_ring_.count(GlobalProcessCtx::Rank()) > 0;
}

Maybe<void> RankGroup::ForEachRank(const std::function<Maybe<void>(int64_t)>& DoEach) const {
  for (int64_t i : ranks_) { JUST(DoEach(i)); }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
