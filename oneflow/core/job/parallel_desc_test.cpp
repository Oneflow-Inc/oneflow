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
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {
namespace test {

TEST(ParallelDesc, continuous_1n4d) {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
}

TEST(ParallelDesc, discrete_1n4d) {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-1");
  parallel_conf.add_device_name("0:2-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 4);
}

TEST(ParallelDesc, continuous_2n8d) {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  parallel_conf.add_device_name("1:0-3");
  ParallelDesc parallel_desc(parallel_conf);
  ASSERT_EQ(parallel_desc.device_tag(), "cpu");
  ASSERT_EQ(parallel_desc.parallel_num(), 8);
}

TEST(ParallelDesc, discrete_2n8d) {
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

}  // namespace test
}  // namespace oneflow
