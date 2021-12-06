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
#ifndef ONEFLOW_CORE_HARDWARE_TOPOLOGY_DESCRIPTOR_H_
#define ONEFLOW_CORE_HARDWARE_TOPOLOGY_DESCRIPTOR_H_

#include <string>
#include <memory>
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace hardware {

class TopologyCPUAffinityDescriptor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TopologyCPUAffinityDescriptor);
  TopologyCPUAffinityDescriptor() = default;
  virtual ~TopologyCPUAffinityDescriptor() = default;
};

class TopologyMemoryAffinityDescriptor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TopologyMemoryAffinityDescriptor);
  TopologyMemoryAffinityDescriptor() = default;
  virtual ~TopologyMemoryAffinityDescriptor() = default;
};

class TopologyDescriptor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TopologyDescriptor);
  TopologyDescriptor() = default;
  virtual ~TopologyDescriptor() = default;

  virtual std::shared_ptr<const TopologyCPUAffinityDescriptor> GetCPUAffinity() const = 0;
  virtual std::shared_ptr<const TopologyMemoryAffinityDescriptor> GetMemoryAffinity() const = 0;
  virtual std::shared_ptr<const TopologyCPUAffinityDescriptor> GetCPUAffinityByPCIBusID(
      const std::string& bus_id) const = 0;
  virtual std::shared_ptr<const TopologyMemoryAffinityDescriptor> GetMemoryAffinityByPCIBusID(
      const std::string& bus_id) const = 0;
  virtual void SetCPUAffinity(
      const std::shared_ptr<const TopologyCPUAffinityDescriptor>& affinity) const = 0;
  virtual void SetMemoryAffinity(
      const std::shared_ptr<const TopologyMemoryAffinityDescriptor>& affinity) const = 0;
  virtual void SetCPUAffinityByPCIBusID(const std::string& bus_id) const;
  virtual void SetMemoryAffinityByPCIBusID(const std::string& bus_id) const;
};

}  // namespace hardware

}  // namespace oneflow

#endif  // ONEFLOW_CORE_HARDWARE_TOPOLOGY_DESCRIPTOR_H_
