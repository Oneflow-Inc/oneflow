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
#include <sstream>
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {

const std::unordered_set<std::string> Device::type_supported({"cuda", "cpu"});

namespace {

inline size_t HashDevice(const std::string& type, int64_t device_id) {
  return std::hash<std::string>()(type) ^ std::hash<int64_t>()(device_id);
}

inline Maybe<MemoryCase> MakeMemoryCase(const std::string& type, int64_t device_id) {
  DeviceType device_type = JUST(DeviceType4DeviceTag(type));
  return MemoryCaseUtil::MakeMemCase(device_type, device_id);
}
}  // namespace

const std::shared_ptr<const ParallelDesc>& Device::parallel_desc_ptr() const {
  return Global<EnvGlobalObjectsScope>::Get()->MutParallelDesc4Device(*this);
}

Device::Device(const std::string& type, int64_t device_id)
    : type_(type), device_id_(device_id), hash_value_(HashDevice(type, device_id)) {}

/*static*/ Maybe<Device> Device::New(const std::string& type, int64_t device_id) {
  std::shared_ptr<Device> device(new Device(type, device_id));
  JUST(device->Init());
  return device;
}

Maybe<void> Device::Init() {
  mem_case_ = JUST(MakeMemoryCase(of_type(), device_id()));
  return Maybe<void>::Ok();
}

std::string Device::of_type() const {
  if (type_ == "cuda") {
    return "gpu";
  } else {
    return type_;
  }
}

std::string Device::ToString() const {
  std::stringstream ss;
  ss << "device(type='";
  ss << type_;
  ss << "', index=";
  ss << device_id_;
  ss << ")";
  return ss.str();
}

Maybe<const ParallelDesc> Device::MakeParallelDescByDevice(const Device& device) {
  int64_t machine_id = GlobalProcessCtx::Rank() / GlobalProcessCtx::NumOfProcessPerNode();
  int64_t device_id = device.device_id();
  std::string machine_device_id = std::to_string(machine_id) + ":" + std::to_string(device_id);
  return std::const_pointer_cast<const ParallelDesc>(
      JUST(ParallelDesc::New(device.of_type(), {machine_device_id}, nullptr)));
}

Maybe<const Device> Device::MakeDeviceByParallelDesc(const ParallelDesc& parallel_desc) {
  std::string type = parallel_desc.device_tag();
  if (parallel_desc.device_tag() == "gpu") { type = "cuda"; }
  std::vector<std::string> machine_device_ids;
  for (const auto& item : parallel_desc.parallel_conf().device_name()) {
    machine_device_ids.emplace_back(item);
  }
  CHECK_EQ_OR_RETURN(machine_device_ids.size(), 1);
  const std::string& machine_device_id = machine_device_ids.at(0);
  size_t pos = machine_device_id.find(':');
  CHECK_NE_OR_RETURN(pos, std::string::npos) << "device_name: " << machine_device_id;
  std::string device_id = machine_device_id.substr(pos + 1);
  CHECK_EQ_OR_RETURN(device_id.find('-'), std::string::npos);
  CHECK_OR_RETURN(IsStrInt(device_id));
  std::shared_ptr<const Device> device = JUST(Device::New(type, std::stoi(device_id)));
  return device;
}

}  // namespace oneflow
