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

namespace oneflow {
namespace test {

TEST(ParallelDesc, push_pull_key_4_ranks_simple) {
  InitNumProcessPerNode();
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0");
  parallel_conf.add_device_name("1:1");
  ParallelDesc parallel_desc(parallel_conf);
  const auto& sorted_rank_ranges =
      CHECK_JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(SymbolOf(parallel_desc)));
  ASSERT_EQ(sorted_rank_ranges->rpc_push_pull_key(), "0-1");
  DestroyNumProcessPerNode();
}

TEST(ParallelDesc, push_pull_key_4_ranks) {
  InitNumProcessPerNode();
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0");
  parallel_conf.add_device_name("1:1");
  parallel_conf.add_device_name("3:3");
  parallel_conf.add_device_name("4:4");
  ParallelDesc parallel_desc(parallel_conf);
  const auto& sorted_rank_ranges =
      CHECK_JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(SymbolOf(parallel_desc)));
  ASSERT_EQ(sorted_rank_ranges->rpc_push_pull_key(), "0-1,3-4");
  DestroyNumProcessPerNode();
}

}  // namespace test
}  // namespace oneflow
