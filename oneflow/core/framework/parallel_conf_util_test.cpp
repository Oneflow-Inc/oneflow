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
#include "oneflow/core/framework/parallel_conf_util.h"

namespace oneflow {
namespace test {

TEST(ParallelConfUtil, MakeParallelConfSuccess) {
  std::string device_tag = "cpu";
  std::vector<std::string> machine_device_ids;
  machine_device_ids.emplace_back("0:0-3");
  machine_device_ids.emplace_back("1:0-3");
  auto parallel_conf = CHECK_JUST(MakeParallelConf(device_tag, machine_device_ids, nullptr));
  ASSERT_EQ(parallel_conf->device_tag(), "cpu");
  ASSERT_EQ(parallel_conf->device_name().size(), 2);
  ASSERT_EQ(parallel_conf->has_hierarchy(), false);
}

TEST(ParallelConfUtil, MakeParallelConfError) {
  std::string device_tag = "cpu";
  std::vector<std::string> machine_device_ids;
  machine_device_ids.emplace_back("0:0-3");
  machine_device_ids.emplace_back("1:0-");
  auto parallel_conf = TRY(MakeParallelConf(device_tag, machine_device_ids, nullptr));
  ASSERT_EQ(parallel_conf.error()->has_check_failed_error(), true);
}

TEST(ParallelConfUtil, GetDeviceTagAndMachineDeviceIdsAndHierarchy) {
  std::shared_ptr<cfg::ParallelConf> parallel_conf = std::make_shared<cfg::ParallelConf>();
  parallel_conf->set_device_tag("cpu");
  parallel_conf->add_device_name("0:0-1");
  parallel_conf->add_device_name("0:2-3");
  parallel_conf->add_device_name("1:0-1");
  parallel_conf->add_device_name("1:2-3");
  parallel_conf->mutable_hierarchy()->add_dim(2);
  parallel_conf->mutable_hierarchy()->add_dim(4);
  std::tuple<std::string, std::vector<std::string>, std::shared_ptr<cfg::ShapeProto>>
      tag_and_dev_ids_and_hierarchy =
          *CHECK_JUST(GetDeviceTagAndMachineDeviceIdsAndHierarchy(parallel_conf));
  std::string device_tag = std::get<0>(tag_and_dev_ids_and_hierarchy);
  std::vector<std::string> machine_device_ids = std::get<1>(tag_and_dev_ids_and_hierarchy);
  std::shared_ptr<cfg::ShapeProto> hierarchy = std::get<2>(tag_and_dev_ids_and_hierarchy);
  ASSERT_EQ(device_tag, "cpu");
  ASSERT_NE(std::count(machine_device_ids.begin(), machine_device_ids.end(), "0:0-1"), 0);
  ASSERT_NE(std::count(machine_device_ids.begin(), machine_device_ids.end(), "0:2-3"), 0);
  ASSERT_NE(std::count(machine_device_ids.begin(), machine_device_ids.end(), "1:0-1"), 0);
  ASSERT_NE(std::count(machine_device_ids.begin(), machine_device_ids.end(), "1:2-3"), 0);
  ASSERT_EQ(std::count(machine_device_ids.begin(), machine_device_ids.end(), "2:0-3"), 0);
  ASSERT_EQ(hierarchy->dim(0), 2);
  ASSERT_EQ(hierarchy->dim(1), 4);
}

}  // namespace test
}  // namespace oneflow
