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
#include "oneflow/core/device/node_device_descriptor.h"
#include "oneflow/core/device/device_descriptor_class.h"
#include "oneflow/core/common/str_util.h"
#include <json.hpp>

namespace oneflow {

namespace device {

namespace {

constexpr char kJsonKeyClasses[] = "classes";
constexpr char kJsonKeyClassName[] = "class_name";
constexpr char kJsonKeySerializedDescriptorList[] = "serialized_descriptor_list";
constexpr char kJsonKeyHostMemorySize[] = "host_memory_size_bytes";

}  // namespace

struct NodeDeviceDescriptor::Impl {
  std::unordered_map<std::string, std::shared_ptr<const DeviceDescriptorList>>
      class_name2descriptor_list;
  size_t host_memory_size_bytes{};
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
  CHECK(it != impl_->class_name2descriptor_list.end());
  return it->second;
}

size_t NodeDeviceDescriptor::HostMemorySizeBytes() const { return impl_->host_memory_size_bytes; }

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
  *serialized = json_object.dump();
}

void NodeDeviceDescriptor::DumpSummary(const std::string& path) const {
  std::string classes_base = JoinPath(path, "classes");
  for (const auto& pair : impl_->class_name2descriptor_list) {
    auto clz = DeviceDescriptorClass::GetRegisteredClass(pair.first);
    CHECK(clz);
    clz->DumpDeviceDescriptorListSummary(pair.second, JoinPath(classes_base, pair.first));
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
  return std::shared_ptr<const NodeDeviceDescriptor>(desc);
}

}  // namespace device

}  // namespace oneflow
