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
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include "oneflow/api/python/utils/tensor_utils.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/ep/include/device.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

static py::object Device_memoryStats(int device) {
  // THPUtils_assert(
  //     THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  // const int device = (int) utils_unpackLong(device_id);

  using oneflow::CUDACachingAllocator::DeviceStats;
  using oneflow::CUDACachingAllocator::Stat;
  using oneflow::CUDACachingAllocator::StatArray;
  using oneflow::CUDACachingAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)> statTypeNames = {
        "all", "small_pool", "large_pool"};
    py::dict dict;
    std::vector<int32_t> stat_len(statTypeNames.size());
    std::iota(stat_len.begin(), stat_len.end(), 0);
    for (const auto i : stat_len) { dict[statTypeNames[i]] = statToDict(statArray[i]); }
    return dict;
  };

  const DeviceStats stats = oneflow::CUDACachingAllocator::GetCUDADeviceStatus(device);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);
  return result;
}

}  // namespace one

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<Device>, std::shared_ptr<Symbol<Device>>>(m, "device")
      .def(py::init([](const std::string& type_or_type_with_device_id) {
        return Device::ParseAndNew(type_or_type_with_device_id).GetOrThrow();
      }))
      .def(py::init([](const std::string& type, int64_t index) {
             return Device::New(type, index).GetOrThrow();
           }),
           py::arg("type"), py::arg("index"))
      .def(py::init([](const Symbol<Device>& other_device) { return other_device; }))
      .def_property_readonly("type", [](const Symbol<Device>& d) { return d->type(); })
      .def_property_readonly("index", [](const Symbol<Device>& d) { return d->device_id(); })
      .def_property_readonly("rematable", [](const Symbol<Device>& d) { return d->rematable(); })
      .def("__str__", [](const Symbol<Device>& d) { return d->ToString(); })
      .def("__repr__", [](const Symbol<Device>& d) { return d->ToRepr(); })
      .def(py::self == py::self)
      .def(py::hash(py::self));

  m.def(
      "max_alignment_size", []() { return ep::kMaxAlignmentRequirement; },
      py::return_value_policy::copy);
  m.def("_cuda_memoryStat", &one::Device_memoryStats);
}

}  // namespace oneflow
