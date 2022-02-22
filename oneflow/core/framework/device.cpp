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
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

const std::unordered_set<std::string> Device::type_supported({"cuda", "cpu"});

namespace {

inline size_t HashDevice(const std::string& type, int64_t device_id) {
  return std::hash<std::string>()(type) ^ std::hash<int64_t>()(device_id);
}

void CheckDeviceType(const std::string& type) {
  if (Device::type_supported.find(type) == Device::type_supported.end()) {
    std::string error_msg =
        "Expected one of cpu, cuda device type at start of device string " + type;
    throw std::runtime_error(error_msg);
  }
}

}  // namespace

Device::Device(const std::string& type, int64_t device_id)
    : type_(type),
      enum_type_(kInvalidDevice),
      device_id_(device_id),
      hash_value_(HashDevice(type, device_id)),
      transport_local_dep_object_(),
      schedule_local_dep_object_(nullptr) {}

Maybe<void> Device::Init() {
  if (type_ == "auto") { return Maybe<void>::Ok(); }
  enum_type_ = JUST(DeviceType4DeviceTag(JUST(of_type())));
  mem_case_ = MemoryCaseUtil::MakeMemCase(enum_type_, device_id_);
  const auto& opt_device_transport_tag = JUST(GetSharedTransportDeviceType());
  if (opt_device_transport_tag.has_value()) {
    const auto& device_transport_tag = *JUST(opt_device_transport_tag);
    transport_local_dep_object_ = GetStaticLocalDepObject4Device(Device(device_transport_tag, 0));
  }
  const auto& schedule_device_type = JUST(GetSharedScheduleDeviceType());
  schedule_local_dep_object_ =
      GetStaticLocalDepObject4Device(Device(schedule_device_type, device_id_));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> Device::New(const std::string& type, int64_t device_id) {
  return ThreadLocalGetOrNew(type, device_id);
}

/* static */ Maybe<Symbol<Device>> Device::ThreadLocalGetOrNew(const std::string& type,
                                                               int64_t device_id) {
  CHECK_GE_OR_RETURN(device_id, 0);
  static thread_local HashMap<std::string, HashMap<int64_t, Symbol<Device>>> map;
  auto* device_id2symbol = &map[type];
  auto iter = device_id2symbol->find(device_id);
  if (iter == device_id2symbol->end()) {
    Device device(type, device_id);
    JUST(device.Init());
    iter = device_id2symbol->emplace(device_id, SymbolOf(device)).first;
  }
  return iter->second;
}

/* static */ Maybe<Symbol<Device>> Device::New(const std::string& type) {
  return New(type, GlobalProcessCtx::LocalRank());
}

/* static */ Maybe<Symbol<Device>> Device::ParseAndNew(
    const std::string& type_or_type_with_device_id) {
  std::string type;
  int device_id = -1;
  JUST(ParsingDeviceTag(type_or_type_with_device_id, &type, &device_id));
  CheckDeviceType(type);
  if (device_id == -1) {
    return Device::New(type);
  } else {
    return Device::New(type, device_id);
  }
}

Maybe<const std::string&> Device::of_type() const {
  static const HashMap<std::string, std::string> type2device_tag{
      {"cpu", "cpu"},
      {"gpu", "gpu"},
      {"cuda", "gpu"},
      {"cuda_h2d", "gpu"},
      {"cuda_d2h", "gpu"},
      {"comm_net", "cpu"},
      {"sync_launched_nccl", "gpu"},
      {"async_launched_nccl", "gpu"},
      {"critical_section", "cpu"},
      {"auto", "auto"},  // Only used for auto generator currently.
  };
  return MapAt(type2device_tag, type());
}

Maybe<const Optional<std::string>&> Device::GetSharedTransportDeviceType() const {
  // share LocalDepObject between sync_launched_nccl and async_launched_nccl
  static const HashMap<std::string, Optional<std::string>> type2type_for_shared_local_dep_object{
      {"cpu", Optional<std::string>()},
      {"gpu", Optional<std::string>()},
      {"cuda", Optional<std::string>()},
      {"cuda_h2d", Optional<std::string>()},
      {"cuda_d2h", Optional<std::string>()},
      {"comm_net", Optional<std::string>()},
      {"sync_launched_nccl", Optional<std::string>("async_launched_nccl")},
      {"async_launched_nccl", Optional<std::string>("async_launched_nccl")},
      {"critical_section", Optional<std::string>()},
  };
  return MapAt(type2type_for_shared_local_dep_object, type());
}

Maybe<const std::string&> Device::GetSharedScheduleDeviceType() const {
  // share LocalDepObject between comm_net and sync_launched_nccl
  static const HashMap<std::string, std::string> type2type_for_shared_local_dep_object{
      {"cpu", "cpu"},
      {"gpu", "cuda"},
      {"cuda", "cuda"},
      {"cuda_h2d", "cuda_h2d"},
      {"cuda_d2h", "cuda_d2h"},
      {"comm_net", "comm_net"},
      {"sync_launched_nccl", "comm_net"},
      {"async_launched_nccl", "async_launched_nccl"},
      {"critical_section", "critical_section"},
  };
  return MapAt(type2type_for_shared_local_dep_object, type());
}

Maybe<const std::string&> GetLocalCallInstructionName(const std::string& type) {
  static const HashMap<std::string, std::string> type2instr_name{
      {"cpu", "cpu.LocalCallOpKernel"},
      {"gpu", "gpu.LocalCallOpKernel"},
      {"cuda", "gpu.LocalCallOpKernel"},
      {"cuda_h2d", "cuda_h2d.LocalCallOpKernel"},
      {"cuda_d2h", "cuda_d2h.LocalCallOpKernel"},
      {"comm_net", "cpu.LocalCallOpKernel"},
      {"sync_launched_nccl", "gpu.LocalCallOpKernel"},
      {"async_launched_nccl", "async.gpu.LocalCallOpKernel"},
      // no compute instruction on critical_section device.
      {"critical_section", "UNIMPLEMENTED INSTRUCTION NAME"},
  };
  return MapAt(type2instr_name, type);
}

Maybe<size_t> Device::instr_local_dep_object_pool_size() const {
  static const size_t kSmallPoolSize = 4;
  static const HashMap<std::string, size_t> type2pool_size{
      {"cpu", GetInstructionHighWaterMark()},
      {"gpu", GetInstructionHighWaterMark()},
      {"cuda", GetInstructionHighWaterMark()},
      {"cuda_h2d", kSmallPoolSize},
      {"cuda_d2h", GetInstructionHighWaterMark()},
      {"comm_net", GetInstructionHighWaterMark()},
      {"sync_launched_nccl", GetInstructionHighWaterMark()},
      {"async_launched_nccl", GetInstructionHighWaterMark()},
  };
  return MapAt(type2pool_size, type());
}

// TODO(jianhao): move this configuration into stream
Maybe<bool> Device::need_soft_sync_stream() const {
  return JUST(local_call_instruction_name()) == "gpu.LocalCallOpKernel";
}

Maybe<const std::string&> Device::local_call_instruction_name() const {
  return GetLocalCallInstructionName(type());
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

std::string Device::ToString() const {
  std::stringstream ss;
  ss << type_;
  ss << ":" << device_id_;
  return ss.str();
}

Maybe<Symbol<Device>> Device::MakeDeviceByParallelDesc(const ParallelDesc& parallel_desc) {
  std::string type = Type4DeviceTag(parallel_desc.device_tag());
  std::vector<std::string> machine_device_ids;
  machine_device_ids.reserve(parallel_desc.parallel_conf().device_name().size());
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
  return Device::New(type, std::stoi(device_id));
}

std::string Device::Type4DeviceTag(const std::string& device_tag) {
  return device_tag == "gpu" ? "cuda" : device_tag;
}

namespace {

Maybe<Symbol<ParallelDesc>> RawGetPlacement(const Device& device) {
  std::string machine_device_id =
      "@" + std::to_string(GlobalProcessCtx::Rank()) + ":" + std::to_string(device.device_id());
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(JUST(device.of_type()));
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

Maybe<void> ParsingDeviceTag(const std::string& device_tag, std::string* device_name,
                             int* device_index) {
  std::string::size_type pos = device_tag.find(':');
  if (pos == std::string::npos) {
    *device_name = device_tag;
    *device_index = -1;
  } else {
    std::string index_str = device_tag.substr(pos + 1);
    CHECK_OR_RETURN(IsStrInt(index_str)) << "Invalid device " << device_tag;
    *device_name = device_tag.substr(0, pos);
    *device_index = std::stoi(index_str);
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
