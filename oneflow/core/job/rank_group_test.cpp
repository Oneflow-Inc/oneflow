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
#include "oneflow/core/control/ctrl_bootstrap.pb.h"

namespace oneflow {
namespace test {

TEST(RankGroup, two_rank) {
  const auto& rank_group = CHECK_JUST(RankGroup::New(std::set<int64_t>{0, 1}));
  int64_t rank = 0;
  rank = CHECK_JUST(rank_group->GetNextRankInRing(0));
  ASSERT_EQ(rank, 1);
  rank = CHECK_JUST(rank_group->GetNextRankInRing(1));
  ASSERT_EQ(rank, 0);
  rank = CHECK_JUST(rank_group->GetPrevRankInRing(0));
  ASSERT_EQ(rank, 1);
  rank = CHECK_JUST(rank_group->GetPrevRankInRing(1));
  ASSERT_EQ(rank, 0);
}

TEST(RankGroup, nonconsecutive_rank) {
  const auto& rank_group = CHECK_JUST(RankGroup::New(std::set<int64_t>{0, 1, 3, 4}));
  int64_t rank = 0;
  rank = CHECK_JUST(rank_group->GetNextRankInRing(0));
  ASSERT_EQ(rank, 1);
  rank = CHECK_JUST(rank_group->GetNextRankInRing(1));
  ASSERT_EQ(rank, 3);
  rank = CHECK_JUST(rank_group->GetNextRankInRing(3));
  ASSERT_EQ(rank, 4);
  rank = CHECK_JUST(rank_group->GetNextRankInRing(4));
  ASSERT_EQ(rank, 0);
  bool is_ok = TRY(rank_group->GetNextRankInRing(2)).IsOk();
  ASSERT_FALSE(is_ok);
  rank = CHECK_JUST(rank_group->GetPrevRankInRing(1));
  ASSERT_EQ(rank, 0);
  rank = CHECK_JUST(rank_group->GetPrevRankInRing(3));
  ASSERT_EQ(rank, 1);
  rank = CHECK_JUST(rank_group->GetPrevRankInRing(4));
  ASSERT_EQ(rank, 3);
  rank = CHECK_JUST(rank_group->GetPrevRankInRing(0));
  ASSERT_EQ(rank, 4);
}

}  // namespace test
}  // namespace oneflow
