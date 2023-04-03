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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {

namespace {

void CheckDeviceType(const std::string& type) {
  if (!TRY(DeviceType4DeviceTag(type)).IsOk()) {
    std::string error_msg = "Expected one of " + PrintAvailableDevices()
                            + " device type at start of device string: " + type;
    throw std::runtime_error(error_msg);
  }
}

}  // namespace

Device::Device(const std::string& type, int64_t device_id)
    : type_(type),
      enum_type_(kInvalidDevice),
      device_id_(device_id),
      hash_value_(Hash(type, device_id)) {}

Maybe<void> Device::Init() {
  if (type_ == "auto") { return Maybe<void>::Ok(); }
  enum_type_ = JUST(DeviceType4DeviceTag(type()));
  {
    DeviceType dev_type = enum_type_;
    if (dev_type == kMockDevice) { dev_type = DeviceType::kCPU; }
    mem_case_ = memory::MakeMemCaseShared(enum_type_, device_id_);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> Device::New(const std::string& type, int64_t device_id) {
  CHECK_GE_OR_RETURN(device_id, 0)
      << Error::InvalidValueError() << "Device ID should be non-negative";
  static thread_local HashMap<std::tuple<std::string, int>, Symbol<Device>> map;
  auto key = std::make_tuple(type, device_id);
  auto iter = map.find(key);
  if (iter == map.end()) {
    Device device(type, device_id);
    JUST(device.Init());
    iter = map.emplace(key, SymbolOf(device)).first;
  }
  return iter->second;
}

/* static */ Maybe<Symbol<Device>> Device::New(const std::string& type) {
  return New(type, GlobalProcessCtx::LocalRank());
}

/* static */ Maybe<Symbol<Device>> Device::ParseAndNew(const std::string& device_str) {
  static thread_local HashMap<std::string, Symbol<Device>> map;
  auto iter = map.find(device_str);
  if (iter == map.end()) {
    auto [type, device_id] = *JUST(ParseDeviceString(device_str));
    CheckDeviceType(type);
    if (device_id == -1) { device_id = GlobalProcessCtx::LocalRank(); }
    Device device(type, device_id);
    JUST(device.Init());
    iter = map.emplace(device_str, SymbolOf(device)).first;
  }
  return iter->second;
}

std::string Device::ToRepr() const {
  std::stringstream ss;
  ss << "device(type='";
  ss << type_;
  ss << "', index=";
  ss << device_id_;
  ss << ")";
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, Symbol<Device> device) {
  os << device->ToRepr();
  return os;
}

std::string Device::ToString() const {
  std::stringstream ss;
  ss << type_;
  ss << ":" << device_id_;
  return ss.str();
}

Maybe<Symbol<Device>> Device::MakeDeviceByParallelDesc(const ParallelDesc& parallel_desc) {
  const std::string& type = parallel_desc.device_tag();
  std::vector<std::string> machine_device_ids;
  machine_device_ids.reserve(parallel_desc.parallel_conf().device_name().size());
  for (const auto& item : parallel_desc.parallel_conf().device_name()) {
    machine_device_ids.emplace_back(item);
  }
  CHECK_EQ_OR_RETURN(machine_device_ids.size(), 1)
      << Error::InvalidValueError() << "Number of machine device should be one";
  const std::string& machine_device_id = machine_device_ids.at(0);
  size_t pos = machine_device_id.find(':');
  CHECK_NE_OR_RETURN(pos, std::string::npos)
      << Error::InvalidValueError() << "Invalid device ID: " << machine_device_id;
  std::string device_id = machine_device_id.substr(pos + 1);
  CHECK_EQ_OR_RETURN(device_id.find('-'), std::string::npos)
      << Error::InvalidValueError() << "Device ID should be non-negative";
  CHECK_OR_RETURN(IsStrInt(device_id))
      << Error::InvalidValueError() << "Device ID is not integer: " << device_id;
  return Device::New(type, std::stoi(device_id));
}

namespace {

Maybe<Symbol<ParallelDesc>> RawGetPlacement(const Device& device) {
  std::string machine_device_id =
      "@" + std::to_string(GlobalProcessCtx::Rank()) + ":" + std::to_string(device.device_id());
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(device.type());
  parallel_conf.add_device_name(machine_device_id);
  return SymbolOf(ParallelDesc(parallel_conf));
}

Maybe<Symbol<ParallelDesc>> RawPlacement4Device(Symbol<Device> device) {
  return RawGetPlacement(*device);
}

}  // namespace

decltype(Device::GetPlacement) Device::GetPlacement =
    DECORATE(&RawGetPlacement, ThreadLocalCopiable);
decltype(Placement4Device) Placement4Device = DECORATE(&RawPlacement4Device, ThreadLocal);

Maybe<std::pair<std::string, int>> ParseDeviceString(const std::string& device_str) {
  std::string::size_type pos = device_str.find(':');
  if (pos == std::string::npos) {
    return std::make_pair(device_str, -1);
  } else {
    std::string index_str = device_str.substr(pos + 1);
    CHECK_OR_RETURN(IsStrInt(index_str))
        << Error::InvalidValueError() << "Invalid device tag " << device_str;
    return std::make_pair(device_str.substr(0, pos), std::stoi(index_str));
  }
}

}  // namespace oneflow
