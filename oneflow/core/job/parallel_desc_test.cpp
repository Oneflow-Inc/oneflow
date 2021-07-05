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
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"

namespace oneflow {

namespace test {

namespace {

void InitRankInfoInCluster(int64_t machine_num) {
  Global<RankInfoInCluster>::New();
  for (size_t i = 0; i < machine_num; ++i) {
    Global<RankInfoInCluster>::Get()->mutable_num_process_distribution()->add_num_process(1);
    (*Global<RankInfoInCluster>::Get()->mutable_rank2node_id())[i] = i;
    (*Global<RankInfoInCluster>::Get()->mutable_node_id2rankoffset())[i] = i;
  }
}

void DestroyRankInfoInCluster() { Global<RankInfoInCluster>::Delete(); }

}  // namespace

TEST(ParallelDesc, continuous_1n4d) {
  InitRankInfoInCluster(1);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
  DestroyRankInfoInCluster();
}

TEST(ParallelDesc, continuous_1n4d_multi_process) {
  InitRankInfoInCluster(1);
  Global<RankInfoInCluster>::Get()->mutable_num_process_distribution()->clear_num_process();
  Global<RankInfoInCluster>::Get()->mutable_num_process_distribution()->add_num_process(4);
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
  DestroyRankInfoInCluster();
}

TEST(ParallelDesc, continuous_1n4d_multi_process_with_rank) {
  InitRankInfoInCluster(1);
  Global<RankInfoInCluster>::Get()->mutable_num_process_distribution()->clear_num_process();
  Global<RankInfoInCluster>::Get()->mutable_num_process_distribution()->add_num_process(4);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("@0:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  const std::vector<int64_t>& machine_ids = parallel_desc.sorted_machine_ids();
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
  ASSERT_EQ(machine_ids.size(), 1);
  ASSERT_EQ(std::count(machine_ids.begin(), machine_ids.end(), 0), 1);
  DestroyRankInfoInCluster();
}

TEST(ParallelDesc, discrete_1n4d) {
  InitRankInfoInCluster(1);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-1");
  parallel_conf.add_device_name("0:2-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
  DestroyRankInfoInCluster();
}

TEST(ParallelDesc, continuous_2n8d) {
  InitRankInfoInCluster(2);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  parallel_conf.add_device_name("1:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 8);
  DestroyRankInfoInCluster();
}

TEST(ParallelDesc, discrete_2n8d) {
  InitRankInfoInCluster(2);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-1");
  parallel_conf.add_device_name("0:2-3");
  parallel_conf.add_device_name("1:0-1");
  parallel_conf.add_device_name("1:2-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 8);
  DestroyRankInfoInCluster();
}

}  // namespace test
}  // namespace oneflow
