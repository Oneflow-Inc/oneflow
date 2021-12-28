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
#include "oneflow/core/hardware/node_device_descriptor.h"
#include "oneflow/core/hardware/device_descriptor_class.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "nlohmann/json.hpp"
#ifdef WITH_HWLOC
#include <hwloc.h>
#endif  // WITH_HWLOC

namespace oneflow {

namespace hardware {

namespace {

constexpr char kJsonKeyClasses[] = "classes";
constexpr char kJsonKeyClassName[] = "class_name";
constexpr char kJsonKeySerializedDescriptorList[] = "serialized_descriptor_list";
constexpr char kJsonKeyHostMemorySize[] = "host_memory_size_bytes";
constexpr char kJsonKeyTopology[] = "topology";

class DummyCPUAffinityDescriptor : public TopologyCPUAffinityDescriptor {
 public:
  DummyCPUAffinityDescriptor() = default;
  ~DummyCPUAffinityDescriptor() override = default;
};

class DummyMemoryAffinityDescriptor : public TopologyMemoryAffinityDescriptor {
 public:
  DummyMemoryAffinityDescriptor() = default;
  ~DummyMemoryAffinityDescriptor() override = default;
};

class DummyTopologyDescriptor : public TopologyDescriptor {
 public:
  DummyTopologyDescriptor() = default;
  ~DummyTopologyDescriptor() override = default;

  std::shared_ptr<const TopologyCPUAffinityDescriptor> GetCPUAffinity() const override {
    return std::make_shared<const DummyCPUAffinityDescriptor>();
  }

  std::shared_ptr<const TopologyMemoryAffinityDescriptor> GetMemoryAffinity() const override {
    return std::make_shared<const DummyMemoryAffinityDescriptor>();
  }

  std::shared_ptr<const TopologyCPUAffinityDescriptor> GetCPUAffinityByPCIBusID(
      const std::string& bus_id) const override {
    return std::make_shared<const DummyCPUAffinityDescriptor>();
  }

  std::shared_ptr<const TopologyMemoryAffinityDescriptor> GetMemoryAffinityByPCIBusID(
      const std::string& bus_id) const override {
    return std::make_shared<const DummyMemoryAffinityDescriptor>();
  }

  void SetCPUAffinity(
      const std::shared_ptr<const TopologyCPUAffinityDescriptor>& affinity) const override {}

  void SetMemoryAffinity(
      const std::shared_ptr<const TopologyMemoryAffinityDescriptor>& affinity) const override {}
};

#ifdef WITH_HWLOC

class HWLocCPUAffinityDescriptor : public TopologyCPUAffinityDescriptor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HWLocCPUAffinityDescriptor);
  explicit HWLocCPUAffinityDescriptor(hwloc_cpuset_t hwloc_cpu_set)
      : hwloc_cpu_set_(hwloc_cpu_set) {}
  ~HWLocCPUAffinityDescriptor() override { hwloc_bitmap_free(hwloc_cpu_set_); }

  hwloc_cpuset_t HWLocCPUSet() const { return hwloc_cpu_set_; }

 private:
  hwloc_cpuset_t hwloc_cpu_set_;
};

class HWLocMemoryAffinityDescriptor : public TopologyMemoryAffinityDescriptor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HWLocMemoryAffinityDescriptor);
  explicit HWLocMemoryAffinityDescriptor(hwloc_bitmap_t hwloc_bitmap, hwloc_membind_policy_t policy)
      : hwloc_bitmap_(hwloc_bitmap), policy_(policy) {}
  ~HWLocMemoryAffinityDescriptor() override { hwloc_bitmap_free(hwloc_bitmap_); }

  hwloc_bitmap_t HWLocBitmap() const { return hwloc_bitmap_; }
  hwloc_membind_policy_t HWLocPolicy() const { return policy_; }

 private:
  hwloc_bitmap_t hwloc_bitmap_;
  hwloc_membind_policy_t policy_;
};

class HWLocTopologyDescriptor : public TopologyDescriptor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HWLocTopologyDescriptor);
  ~HWLocTopologyDescriptor() override { hwloc_topology_destroy(topology_); }

  std::shared_ptr<const TopologyCPUAffinityDescriptor> GetCPUAffinity() const override {
    hwloc_bitmap_t set = hwloc_bitmap_alloc();
    if (hwloc_get_cpubind(topology_, set, HWLOC_CPUBIND_THREAD) != 0) { return nullptr; }
    return std::make_shared<const HWLocCPUAffinityDescriptor>(set);
  }

  std::shared_ptr<const TopologyMemoryAffinityDescriptor> GetMemoryAffinity() const override {
    hwloc_bitmap_t set = hwloc_bitmap_alloc();
    hwloc_membind_policy_t policy{};
    if (hwloc_get_membind(topology_, set, &policy, HWLOC_MEMBIND_THREAD) != 0) { return nullptr; }
    return std::make_shared<const HWLocMemoryAffinityDescriptor>(set, policy);
  }

  std::shared_ptr<const TopologyCPUAffinityDescriptor> GetCPUAffinityByPCIBusID(
      const std::string& bus_id) const override {
    if (bus_id.empty()) { return nullptr; }
    hwloc_obj_t non_io_ancestor = GetNonIOAncestorByPCIBusID(bus_id);
    if (non_io_ancestor == nullptr) { return nullptr; }
    if (non_io_ancestor->cpuset == nullptr) { return nullptr; }
    return std::make_shared<const HWLocCPUAffinityDescriptor>(
        hwloc_bitmap_dup(non_io_ancestor->cpuset));
  }

  std::shared_ptr<const TopologyMemoryAffinityDescriptor> GetMemoryAffinityByPCIBusID(
      const std::string& bus_id) const override {
    if (bus_id.empty()) { return nullptr; }
    hwloc_obj_t non_io_ancestor = GetNonIOAncestorByPCIBusID(bus_id);
    if (non_io_ancestor == nullptr) { return nullptr; }
    if (non_io_ancestor->cpuset == nullptr) { return nullptr; }
    return std::make_shared<const HWLocMemoryAffinityDescriptor>(
        hwloc_bitmap_dup(non_io_ancestor->cpuset), HWLOC_MEMBIND_BIND);
  }

  void SetCPUAffinity(
      const std::shared_ptr<const TopologyCPUAffinityDescriptor>& affinity) const override {
    auto hwloc_affinity = std::dynamic_pointer_cast<const HWLocCPUAffinityDescriptor>(affinity);
    if (!hwloc_affinity) { return; }
    hwloc_set_cpubind(topology_, hwloc_affinity->HWLocCPUSet(), HWLOC_CPUBIND_THREAD);
  }

  void SetMemoryAffinity(
      const std::shared_ptr<const TopologyMemoryAffinityDescriptor>& affinity) const override {
    auto hwloc_affinity = std::dynamic_pointer_cast<const HWLocMemoryAffinityDescriptor>(affinity);
    if (!hwloc_affinity) { return; }
    hwloc_set_membind(topology_, hwloc_affinity->HWLocBitmap(), hwloc_affinity->HWLocPolicy(),
                      HWLOC_MEMBIND_THREAD);
  }

  static std::shared_ptr<const HWLocTopologyDescriptor> Query() {
    hwloc_topology_t topology = nullptr;
    do {
      if (hwloc_topology_init(&topology) != 0) { break; }
      if (hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL) != 0) { break; }
      if (hwloc_topology_load(topology) != 0) { break; }
      auto* desc = new HWLocTopologyDescriptor(topology);
      return std::shared_ptr<const HWLocTopologyDescriptor>(desc);
    } while (false);
    if (topology != nullptr) { hwloc_topology_destroy(topology); }
    return nullptr;
  }

  static std::shared_ptr<const HWLocTopologyDescriptor> Deserialize(const std::string& serialized) {
    hwloc_topology_t topology = nullptr;
    do {
      if (hwloc_topology_init(&topology) != 0) { break; }
      if (hwloc_topology_set_xmlbuffer(topology, serialized.data(),
                                       static_cast<int>(serialized.size()))
          != 0) {
        break;
      }
      if (hwloc_topology_load(topology) != 0) { break; }
      auto* desc = new HWLocTopologyDescriptor(topology);
      return std::shared_ptr<const HWLocTopologyDescriptor>(desc);
    } while (false);
    if (topology != nullptr) { hwloc_topology_destroy(topology); }
    return nullptr;
  }

  void Serialize(std::string* serialized) const {
    char* buffer = nullptr;
    int len = 0;
    if (hwloc_topology_export_xmlbuffer(topology_, &buffer, &len, 0) == 0) {
      *serialized = buffer;
      hwloc_free_xmlbuffer(topology_, buffer);
    }
  }

 private:
  hwloc_obj_t GetNonIOAncestorByPCIBusID(const std::string& pci_bus_id) const {
    hwloc_obj_t device = hwloc_get_pcidev_by_busidstring(topology_, pci_bus_id.data());
    if (device == nullptr) { return nullptr; }
    hwloc_obj_t non_io_ancestor = hwloc_get_non_io_ancestor_obj(topology_, device);
    return non_io_ancestor;
  }

  explicit HWLocTopologyDescriptor(hwloc_topology_t topology) : topology_(topology) {}
  hwloc_topology_t topology_{};
};

#endif  // WITH_HWLOC

std::shared_ptr<const TopologyDescriptor> QueryTopologyDescriptor() {
  std::shared_ptr<const TopologyDescriptor> topology;
#ifdef WITH_HWLOC
  topology = HWLocTopologyDescriptor::Query();
#endif  // WITH_HWLOC
  if (!topology) { topology.reset(new DummyTopologyDescriptor()); }
  return topology;
}

std::shared_ptr<const TopologyDescriptor> DeserializeTopologyDescriptor(
    const std::string& serialized) {
  std::shared_ptr<const TopologyDescriptor> topology;
  if (serialized.empty()) {
    topology.reset(new DummyTopologyDescriptor());
  } else {
#ifdef WITH_HWLOC
    topology = HWLocTopologyDescriptor::Deserialize(serialized);
#else
    UNIMPLEMENTED();
#endif  // WITH_HWLOC
  }
  if (!topology) { topology.reset(new DummyTopologyDescriptor()); }
  return topology;
}

void SerializeTopologyDescriptor(const std::shared_ptr<const TopologyDescriptor>& topology,
                                 std::string* serialized) {
#ifdef WITH_HWLOC
  auto hwloc_topology = std::dynamic_pointer_cast<const HWLocTopologyDescriptor>(topology);
  if (hwloc_topology) { hwloc_topology->Serialize(serialized); }
#endif  // WITH_HWLOC
}

}  // namespace

struct NodeDeviceDescriptor::Impl {
  std::unordered_map<std::string, std::shared_ptr<const DeviceDescriptorList>>
      class_name2descriptor_list;
  size_t host_memory_size_bytes{};
  std::shared_ptr<const TopologyDescriptor> topology;
};

NodeDeviceDescriptor::NodeDeviceDescriptor() { impl_.reset(new Impl()); }

NodeDeviceDescriptor::~NodeDeviceDescriptor() = default;

bool NodeDeviceDescriptor::HasDeviceClass(const std::string& class_name) const {
  return impl_->class_name2descriptor_list.find(class_name)
         != impl_->class_name2descriptor_list.end();
}

std::shared_ptr<const DeviceDescriptorList> NodeDeviceDescriptor::GetDeviceDescriptorList(
    const std::string& class_name) const {
  auto it = impl_->class_name2descriptor_list.find(class_name);
  if (it != impl_->class_name2descriptor_list.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

std::shared_ptr<const DeviceDescriptor> NodeDeviceDescriptor::GetDevice(
    const std::string& class_name, size_t ordinal) const {
  const auto device_list = GetDeviceDescriptorList(class_name);
  if (device_list) {
    return device_list->GetDevice(ordinal);
  } else {
    return nullptr;
  }
}

size_t NodeDeviceDescriptor::HostMemorySizeBytes() const { return impl_->host_memory_size_bytes; }

std::shared_ptr<const TopologyDescriptor> NodeDeviceDescriptor::Topology() const {
  return impl_->topology;
}

void NodeDeviceDescriptor::Serialize(std::string* serialized) const {
  nlohmann::json json_object;
  json_object[kJsonKeyHostMemorySize] = impl_->host_memory_size_bytes;
  for (const auto& pair : impl_->class_name2descriptor_list) {
    std::string serialized_descriptor_list;
    auto clz = DeviceDescriptorClass::GetRegisteredClass(pair.first);
    CHECK(clz);
    clz->SerializeDeviceDescriptorList(pair.second, &serialized_descriptor_list);
    json_object[kJsonKeyClasses].push_back(
        {{kJsonKeyClassName, clz->Name()},
         {kJsonKeySerializedDescriptorList, serialized_descriptor_list}});
  }
  std::string serialized_topology;
  SerializeTopologyDescriptor(impl_->topology, &serialized_topology);
  json_object[kJsonKeyTopology] = serialized_topology;
  *serialized = json_object.dump();
}

void NodeDeviceDescriptor::DumpSummary(const std::string& path) const {
  std::string classes_base = JoinPath(path, "classes");
  for (const auto& pair : impl_->class_name2descriptor_list) {
    auto clz = DeviceDescriptorClass::GetRegisteredClass(pair.first);
    CHECK(clz);
    clz->DumpDeviceDescriptorListSummary(pair.second, JoinPath(classes_base, pair.first));
  }
  std::string serialized_topology;
  SerializeTopologyDescriptor(impl_->topology, &serialized_topology);
  if (!serialized_topology.empty()) {
    TeePersistentLogStream::Create(JoinPath(path, "topology"))->Write(serialized_topology);
  }
}

std::shared_ptr<const NodeDeviceDescriptor> NodeDeviceDescriptor::Query() {
  auto* desc = new NodeDeviceDescriptor();
  desc->impl_->host_memory_size_bytes = GetAvailableCpuMemSize();
  const size_t num_classes = DeviceDescriptorClass::GetRegisteredClassesCount();
  for (size_t i = 0; i < num_classes; ++i) {
    std::shared_ptr<const DeviceDescriptorClass> descriptor_class =
        DeviceDescriptorClass::GetRegisteredClass(i);
    desc->impl_->class_name2descriptor_list.emplace(descriptor_class->Name(),
                                                    descriptor_class->QueryDeviceDescriptorList());
  }
  desc->impl_->topology = QueryTopologyDescriptor();
  return std::shared_ptr<const NodeDeviceDescriptor>(desc);
}

std::shared_ptr<const NodeDeviceDescriptor> NodeDeviceDescriptor::Deserialize(
    const std::string& serialized) {
  auto json_object = nlohmann::json::parse(serialized);
  auto* desc = new NodeDeviceDescriptor();
  desc->impl_->host_memory_size_bytes = json_object[kJsonKeyHostMemorySize];
  auto num_classes = json_object[kJsonKeyClasses].size();
  for (int i = 0; i < num_classes; ++i) {
    const std::string class_name = json_object[kJsonKeyClasses].at(i)[kJsonKeyClassName];
    const std::string serialized_descriptor_list =
        json_object[kJsonKeyClasses].at(i)[kJsonKeySerializedDescriptorList];
    auto clz = DeviceDescriptorClass::GetRegisteredClass(class_name);
    CHECK(clz);
    const auto descriptor_list = clz->DeserializeDeviceDescriptorList(serialized_descriptor_list);
    desc->impl_->class_name2descriptor_list.emplace(class_name, descriptor_list);
  }
  desc->impl_->topology = DeserializeTopologyDescriptor(json_object[kJsonKeyTopology]);
  return std::shared_ptr<const NodeDeviceDescriptor>(desc);
}

}  // namespace hardware

}  // namespace oneflow
