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
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/common/shape.cfg.h"
#include "oneflow/core/common/shape.pb.h"

namespace oneflow {

Maybe<std::tuple<std::string, std::vector<std::string>, std::shared_ptr<cfg::ShapeProto>>>
GetDeviceTagAndMachineDeviceIdsAndHierarchy(
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  std::vector<std::string> machine_device_ids;
  for (const std::string& device_name : parallel_conf->device_name()) {
    machine_device_ids.emplace_back(device_name);
  }
  std::shared_ptr<cfg::ShapeProto> hierarchy;
  if (parallel_conf->has_hierarchy()) {
    hierarchy.reset(new cfg::ShapeProto(parallel_conf->hierarchy()));
  }
  return std::make_tuple(parallel_conf->device_tag(), machine_device_ids, hierarchy);
}

Maybe<cfg::ParallelConf> MakeParallelConf(const std::string& device_tag,
                                          const std::vector<std::string>& machine_device_ids,
                                          const std::shared_ptr<Shape>& hierarchy) {
  std::shared_ptr<cfg::ParallelConf> parallel_conf = std::make_shared<cfg::ParallelConf>();
  parallel_conf->set_device_tag(device_tag);
  for (const std::string& machine_device_id : machine_device_ids) {
    size_t pos = machine_device_id.find(':');
    CHECK_NE_OR_RETURN(pos, std::string::npos) << "device_name: " << machine_device_id;
    std::string machine_id = machine_device_id.substr(0, pos);
    CHECK_OR_RETURN((machine_id[0] == '@' || IsStrInt(machine_id)));
    std::string device_id = machine_device_id.substr(pos + 1);
    size_t minus_pos = device_id.rfind('-');
    if (minus_pos == std::string::npos) {
      CHECK_OR_RETURN(IsStrInt(device_id));
    } else {
      std::string min_id = device_id.substr(0, minus_pos);
      CHECK_OR_RETURN(IsStrInt(min_id));
      std::string max_id = device_id.substr(minus_pos + 1);
      CHECK_OR_RETURN(IsStrInt(max_id));
    }
    parallel_conf->add_device_name(machine_device_id);
    if (hierarchy) {
      ShapeProto proto;
      hierarchy->ToProto(&proto);
      parallel_conf->mutable_hierarchy()->CopyFrom(cfg::ShapeProto(proto));
    }
  }
  return parallel_conf;
}

}  // namespace oneflow
