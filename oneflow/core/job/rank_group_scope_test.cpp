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
#include "gtest/gtest.h"
#include <algorithm>
#include <set>
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"

namespace oneflow {
namespace test {

TEST(RankGroupScope, initial) {
  const auto& rank_group0 = CHECK_JUST(RankGroup::New(std::set<int64_t>{0, 1, 2}));
  auto rank_group_scope = CHECK_JUST(RankGroupScope::MakeInitialRankGroupScope(rank_group0));
  int64_t rank = 0;
  const auto& rank_group = CHECK_JUST(RankGroupScope::CurrentRankGroup());
  rank = CHECK_JUST(rank_group->GetNextRankInRing(0));
  ASSERT_EQ(rank, 1);
  rank_group_scope.reset();
  ASSERT_FALSE(TRY(RankGroupScope::CurrentRankGroup()).IsOk());
}

TEST(RankGroupScope, nonconsecutive_rank) {
  const auto& rank_group0 = CHECK_JUST(RankGroup::New(std::set<int64_t>{0, 1, 2}));
  auto rank_group_scope0 = CHECK_JUST(RankGroupScope::MakeInitialRankGroupScope(rank_group0));
  int64_t rank = 0;
  const auto& rank_group = CHECK_JUST(RankGroupScope::CurrentRankGroup());
  rank = CHECK_JUST(rank_group->GetNextRankInRing(0));
  ASSERT_EQ(rank, 1);
  rank = CHECK_JUST(rank_group->GetNextRankInRing(2));
  ASSERT_EQ(rank, 0);
  {
    const auto& rank_group1 = CHECK_JUST(RankGroup::New(std::set<int64_t>{0, 1}));
    auto rank_group_scope1 = CHECK_JUST(RankGroupScope::MakeNestedRankGroupScope(rank_group1));
    {
      const auto& rank_group2 = CHECK_JUST(RankGroup::New(std::set<int64_t>{0}));
      auto rank_group_scope2 = CHECK_JUST(RankGroupScope::MakeNestedRankGroupScope(rank_group2));
      const auto& current_rank_group = CHECK_JUST(RankGroupScope::CurrentRankGroup());
      ASSERT_TRUE(rank_group2 == current_rank_group);
      const auto& root_rank_group = CHECK_JUST(RankGroupScope::RootRankGroup());
      ASSERT_TRUE(rank_group == root_rank_group);
      rank_group_scope2.reset();
    }
    rank_group_scope1.reset();
  }
  rank_group_scope0.reset();
}

}  // namespace test
}  // namespace oneflow
