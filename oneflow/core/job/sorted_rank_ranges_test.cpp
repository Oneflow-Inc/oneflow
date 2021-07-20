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
#include <algorithm>
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sorted_rank_ranges.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"

namespace oneflow {
namespace test {

namespace {

struct GlobaProcessCtxScope final {
  GlobaProcessCtxScope() {
    Global<ProcessCtx>::New();
    auto* ctx = Global<ProcessCtx>::Get();
    ctx->mutable_ctrl_addr()->Add();
    ctx->set_rank(0);
    ctx->set_node_size(1);
  }
  ~GlobaProcessCtxScope() { Global<ProcessCtx>::Delete(); }
};

}  // namespace

TEST(ParallelDesc, two_rank) {
  GlobaProcessCtxScope scope{};
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0");
  parallel_conf.add_device_name("1:1");
  ParallelDesc parallel_desc(parallel_conf);
  const auto& sorted_rank_ranges =
      CHECK_JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(SymbolOf(parallel_desc)));
  int64_t rank = 0;
  rank = CHECK_JUST(sorted_rank_ranges->GetNextRankInRing(0));
  ASSERT_EQ(rank, 1);
  rank = CHECK_JUST(sorted_rank_ranges->GetNextRankInRing(1));
  ASSERT_EQ(rank, 0);
  rank = CHECK_JUST(sorted_rank_ranges->GetPrevRankInRing(0));
  ASSERT_EQ(rank, 1);
  rank = CHECK_JUST(sorted_rank_ranges->GetPrevRankInRing(1));
  ASSERT_EQ(rank, 0);
}

TEST(ParallelDesc, nonconsecutive_rank) {
  GlobaProcessCtxScope scope{};
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0");
  parallel_conf.add_device_name("1:1");
  parallel_conf.add_device_name("3:3");
  parallel_conf.add_device_name("4:4");
  ParallelDesc parallel_desc(parallel_conf);
  const auto& sorted_rank_ranges =
      CHECK_JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(SymbolOf(parallel_desc)));
  int64_t rank = 0;
  rank = CHECK_JUST(sorted_rank_ranges->GetNextRankInRing(0));
  ASSERT_EQ(rank, 1);
  rank = CHECK_JUST(sorted_rank_ranges->GetNextRankInRing(1));
  ASSERT_EQ(rank, 3);
  rank = CHECK_JUST(sorted_rank_ranges->GetNextRankInRing(3));
  ASSERT_EQ(rank, 4);
  rank = CHECK_JUST(sorted_rank_ranges->GetNextRankInRing(4));
  ASSERT_EQ(rank, 0);
  bool is_ok = TRY(sorted_rank_ranges->GetNextRankInRing(2)).IsOk();
  ASSERT_FALSE(is_ok);
  rank = CHECK_JUST(sorted_rank_ranges->GetPrevRankInRing(1));
  ASSERT_EQ(rank, 0);
  rank = CHECK_JUST(sorted_rank_ranges->GetPrevRankInRing(3));
  ASSERT_EQ(rank, 1);
  rank = CHECK_JUST(sorted_rank_ranges->GetPrevRankInRing(4));
  ASSERT_EQ(rank, 3);
  rank = CHECK_JUST(sorted_rank_ranges->GetPrevRankInRing(0));
  ASSERT_EQ(rank, 4);
}

}  // namespace test
}  // namespace oneflow
