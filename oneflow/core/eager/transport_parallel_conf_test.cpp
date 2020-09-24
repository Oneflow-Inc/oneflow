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
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/eager/transport_parallel_conf.h"

namespace oneflow {
namespace eager {
namespace test {

TEST(TransportParallelConf, transport_instruction_parallel_conf) {
  vm::TestResourceDescScope resource_desc_scope(4, 4, 2);
  std::unique_ptr<ParallelDesc> src;
  {
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name("0:0-3");
    src.reset(new ParallelDesc(parallel_conf));
  }
  std::unique_ptr<ParallelDesc> dst;
  {
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name("0:2-3");
    parallel_conf.add_device_name("1:0-1");
    dst.reset(new ParallelDesc(parallel_conf));
  }
  auto parallel_confs = CHECK_JUST(MakeTransportInstructionParallelConfs(*src, *dst));
  ASSERT_TRUE(parallel_confs->parallel_conf_group_size() == 2);
  ASSERT_TRUE(parallel_confs->parallel_conf_group(0).parallel_conf_size() == 2);
  ASSERT_TRUE(parallel_confs->parallel_conf_group(1).parallel_conf_size() == 2);
  ASSERT_TRUE(parallel_confs->parallel_conf_group(0).parallel_conf_size() == 2);
  ASSERT_TRUE(parallel_confs->parallel_conf_group(1).parallel_conf_size() == 2);
  const auto& parallel_conf0 = parallel_confs->parallel_conf_group(0).parallel_conf(0);
  const auto& parallel_conf1 = parallel_confs->parallel_conf_group(0).parallel_conf(1);
  const auto& parallel_conf2 = parallel_confs->parallel_conf_group(1).parallel_conf(0);
  const auto& parallel_conf3 = parallel_confs->parallel_conf_group(1).parallel_conf(1);
  ASSERT_TRUE(parallel_conf0.device_tag() == "cpu");
  ASSERT_TRUE(parallel_conf1.device_tag() == "cpu");
  ASSERT_TRUE(parallel_conf2.device_tag() == "cpu");
  ASSERT_TRUE(parallel_conf3.device_tag() == "cpu");
  ASSERT_TRUE(parallel_conf0.device_name_size() == 1);
  ASSERT_TRUE(parallel_conf1.device_name_size() == 1);
  ASSERT_TRUE(parallel_conf2.device_name_size() == 1);
  ASSERT_TRUE(parallel_conf3.device_name_size() == 1);
  ASSERT_TRUE(parallel_conf0.device_name(0) == "0:0");
  ASSERT_TRUE(parallel_conf1.device_name(0) == "0:1");
  ASSERT_TRUE(parallel_conf2.device_name(0) == "0:2");
  ASSERT_TRUE(parallel_conf3.device_name(0) == "0:3");
}

}  // namespace test
}  // namespace eager
}  // namespace oneflow
