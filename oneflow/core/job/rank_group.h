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
#ifndef ONEFLOW_CORE_JOB_RANK_GROUP_H_
#define ONEFLOW_CORE_JOB_RANK_GROUP_H_

#include <functional>
#include <vector>
#include <unordered_map>
#include <set>
#include <string>
#include <memory>
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/device_type.h"

namespace oneflow {

class ParallelDesc;

class RankGroup final {
 public:
  ~RankGroup() = default;

  static Maybe<Symbol<RankGroup>> New(const std::set<int64_t>& ranks);
  static Maybe<Symbol<RankGroup>> New(Symbol<ParallelDesc> parallel_desc);
  static Maybe<Symbol<RankGroup>> DefaultRankGroup();

  static Maybe<Symbol<ParallelDesc>> GetDefaultParallelDesc(DeviceType device_type,
                                                            Symbol<RankGroup> rank_group);

  bool operator==(const RankGroup& that) const { return this->ranks_ == that.ranks_; }
  bool operator!=(const RankGroup& that) const { return !(*this == that); }

  size_t size() const { return ranks_.size(); }
  size_t hash_value() const { return hash_value_; }
  Maybe<int64_t> GetNextRankInRing(int64_t rank) const;
  Maybe<int64_t> GetNextRankInRing() const;
  Maybe<int64_t> GetPrevRankInRing(int64_t rank) const;
  Maybe<int64_t> GetPrevRankInRing() const;
  bool ContainingCurrentRank() const;

  Maybe<void> ForEachRank(const std::function<Maybe<void>(int64_t)>&) const;

 private:
  RankGroup() = default;
  Maybe<void> Init(const std::set<int64_t>& ranks);
  static Maybe<Symbol<RankGroup>> RawNew(Symbol<ParallelDesc> parallel_desc);

  std::set<int64_t> ranks_;
  std::unordered_map<int64_t, int64_t> rank2next_rank_in_ring_;
  std::unordered_map<int64_t, int64_t> rank2prev_rank_in_ring_;
  size_t hash_value_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::RankGroup> final {
  size_t operator()(const oneflow::RankGroup& rank_group) const { return rank_group.hash_value(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_JOB_RANK_GROUP_H_
