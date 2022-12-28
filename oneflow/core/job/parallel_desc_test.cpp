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
#include "gtest/gtest.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"

namespace oneflow {

namespace test {

namespace {

struct GlobaProcessCtxScope final {
  GlobaProcessCtxScope(int64_t node_size, int64_t world_size) {
    Singleton<ProcessCtx>::New();
    auto* ctx = Singleton<ProcessCtx>::Get();
    for (int i = 0; i < world_size; ++i) { ctx->mutable_ctrl_addr()->Add(); }
    ctx->set_rank(0);
    ctx->set_node_size(node_size);
  }
  ~GlobaProcessCtxScope() { Singleton<ProcessCtx>::Delete(); }
};

}  // namespace

TEST(ParallelDesc, continuous_1n4d) {
  GlobaProcessCtxScope scope(1, 4);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
}

TEST(ParallelDesc, continuous_1n4d_multi_process) {
  GlobaProcessCtxScope scope(1, 4);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  const std::vector<int64_t>& machine_ids = parallel_desc.sorted_machine_ids();
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
  ASSERT_EQ(std::count(machine_ids.begin(), machine_ids.end(), 0), 1);
  ASSERT_EQ(std::count(machine_ids.begin(), machine_ids.end(), 1), 1);
  ASSERT_EQ(std::count(machine_ids.begin(), machine_ids.end(), 2), 1);
  ASSERT_EQ(std::count(machine_ids.begin(), machine_ids.end(), 3), 1);
}

TEST(ParallelDesc, continuous_1n4d_multi_process_with_rank) {
  GlobaProcessCtxScope scope(1, 4);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("@0:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  const std::vector<int64_t>& machine_ids = parallel_desc.sorted_machine_ids();
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
  ASSERT_EQ(machine_ids.size(), 1);
  ASSERT_EQ(std::count(machine_ids.begin(), machine_ids.end(), 0), 1);
}

TEST(ParallelDesc, discrete_1n4d) {
  GlobaProcessCtxScope scope(1, 4);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-1");
  parallel_conf.add_device_name("0:2-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
}

TEST(ParallelDesc, continuous_2n8d) {
  GlobaProcessCtxScope scope(2, 8);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  parallel_conf.add_device_name("1:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 8);
}

TEST(ParallelDesc, discrete_2n8d) {
  GlobaProcessCtxScope scope(2, 8);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-1");
  parallel_conf.add_device_name("0:2-3");
  parallel_conf.add_device_name("1:0-1");
  parallel_conf.add_device_name("1:2-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 8);
}

TEST(GetBroadcastGroup, naive_1n1d) {
  GlobaProcessCtxScope scope(1, 1);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0");
  const auto& parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
  const auto& map = CHECK_JUST(GetBroadcastGroup(parallel_desc, parallel_desc));
  ASSERT_EQ(map->size(), 1);
  ASSERT_EQ(map->begin()->first, 0);
  ASSERT_TRUE(map->begin()->second == parallel_desc);
}

TEST(GetBroadcastGroup, naive_1n4d) {
  GlobaProcessCtxScope scope(1, 4);
  ParallelConf src_parallel_conf;
  src_parallel_conf.set_device_tag("cpu");
  src_parallel_conf.add_device_name("0:0");
  const auto& src_parallel_desc = SymbolOf(ParallelDesc(src_parallel_conf));
  ParallelConf dst_parallel_conf;
  dst_parallel_conf.set_device_tag("cpu");
  dst_parallel_conf.add_device_name("0:0-3");
  const auto& dst_parallel_desc = SymbolOf(ParallelDesc(dst_parallel_conf));
  const auto& map = CHECK_JUST(GetBroadcastGroup(src_parallel_desc, dst_parallel_desc));
  ASSERT_EQ(map->size(), 4);
  for (int i = 0; i < 4; ++i) {
    const auto& iter = map->find(i);
    ASSERT_TRUE(iter != map->end());
    ASSERT_TRUE(iter->second == dst_parallel_desc);
  }
}

TEST(GetBroadcastGroup, naive_2n8d) {
  GlobaProcessCtxScope scope(2, 8);
  ParallelConf src_parallel_conf;
  src_parallel_conf.set_device_tag("cpu");
  src_parallel_conf.add_device_name("0:0");
  src_parallel_conf.add_device_name("1:0");
  const auto& src_parallel_desc = SymbolOf(ParallelDesc(src_parallel_conf));
  ParallelConf dst_parallel_conf;
  dst_parallel_conf.set_device_tag("cpu");
  dst_parallel_conf.add_device_name("0:0-3");
  dst_parallel_conf.add_device_name("1:0-3");
  const auto& dst_parallel_desc = SymbolOf(ParallelDesc(dst_parallel_conf));
  const auto& map = CHECK_JUST(GetBroadcastGroup(src_parallel_desc, dst_parallel_desc));
  ASSERT_EQ(map->size(), 8);

  ParallelConf first_node_parallel_conf;
  first_node_parallel_conf.set_device_tag("cpu");
  first_node_parallel_conf.add_device_name("0:0-3");
  const auto& first_node_parallel_desc = SymbolOf(ParallelDesc(first_node_parallel_conf));
  for (int i = 0; i < 4; ++i) {
    const auto& iter = map->find(i);
    ASSERT_TRUE(iter != map->end());
    ASSERT_TRUE(iter->second == first_node_parallel_desc);
  }
  ParallelConf second_node_parallel_conf;
  second_node_parallel_conf.set_device_tag("cpu");
  second_node_parallel_conf.add_device_name("1:0-3");
  const auto& second_node_parallel_desc = SymbolOf(ParallelDesc(second_node_parallel_conf));
  for (int i = 4; i < 8; ++i) {
    const auto& iter = map->find(i);
    ASSERT_TRUE(iter != map->end());
    ASSERT_TRUE(iter->second == second_node_parallel_desc);
  }
}

}  // namespace test
}  // namespace oneflow
