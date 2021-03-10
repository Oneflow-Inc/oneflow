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

void InitNumProcessPerNode() {
  Global<NumProcessPerNode>::New();
  Global<NumProcessPerNode>::Get()->set_value(1);
}

void DestroyNumProcessPerNode() { Global<NumProcessPerNode>::Delete(); }

}  // namespace

TEST(ParallelDesc, continuous_1n4d) {
  InitNumProcessPerNode();
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
  DestroyNumProcessPerNode();
}

TEST(ParallelDesc, continuous_1n4d_multi_process) {
  InitNumProcessPerNode();
  Global<NumProcessPerNode>::Get()->set_value(4);
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
  DestroyNumProcessPerNode();
}

TEST(ParallelDesc, discrete_1n4d) {
  InitNumProcessPerNode();
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-1");
  parallel_conf.add_device_name("0:2-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
  DestroyNumProcessPerNode();
}

TEST(ParallelDesc, continuous_2n8d) {
  InitNumProcessPerNode();
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  parallel_conf.add_device_name("1:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 8);
  DestroyNumProcessPerNode();
}

TEST(ParallelDesc, discrete_2n8d) {
  InitNumProcessPerNode();
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-1");
  parallel_conf.add_device_name("0:2-3");
  parallel_conf.add_device_name("1:0-1");
  parallel_conf.add_device_name("1:2-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 8);
  DestroyNumProcessPerNode();
}

}  // namespace test
}  // namespace oneflow
