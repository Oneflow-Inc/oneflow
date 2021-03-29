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
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/profiler/profiler.h"

namespace py = pybind11;

namespace oneflow {

namespace {

struct PlacementSymbolExportUtil {
  static std::shared_ptr<ParallelDesc> ApiCreatePlacementSymbol(
      int64_t symbol_id, const std::shared_ptr<cfg::ParallelConf>& symbol_conf) {
    ParallelConf symbol_pb;
    symbol_conf->ToProto(&symbol_pb);
    return ParallelDesc::New(symbol_id, symbol_pb).GetPtrOrThrow();
  }

  static Maybe<ParallelDesc> CreatePlacementSymbol(
      const std::string& device_tag, const std::vector<std::string>& machine_device_ids,
      const std::shared_ptr<Shape>& hierarchy) {
    const auto parallel_conf =
        MakeParallelConf(device_tag, machine_device_ids, hierarchy).GetPtrOrThrow();
    std::shared_ptr<ParallelDesc> parallel_desc;
    JUST(LogicalRun([&parallel_desc, &parallel_conf](
                        const std::shared_ptr<InstructionsBuilder>& builder) -> Maybe<void> {
      OF_PROFILER_RANGE_PUSH("CallLogicalRun: CreatePlacementSymbol");
      parallel_desc = JUST(builder->GetParallelDescSymbol(parallel_conf));
      OF_PROFILER_RANGE_POP();
      return Maybe<void>::Ok();
    }));
    return parallel_desc;
  }

  static std::shared_ptr<ParallelDesc> ApiCreatePlacementSymbol(
      const std::string& device_tag, const std::vector<std::string>& machine_device_ids,
      const std::shared_ptr<Shape>& hierarchy) {
    return CreatePlacementSymbol(device_tag, machine_device_ids, hierarchy).GetPtrOrThrow();
  }

  static HashMap<int64_t, std::vector<int64_t>> MachineId2DeviceIdList(const ParallelDesc& x) {
    const auto map_with_shared_ptr = x.machine_id2sorted_dev_phy_ids();
    // pybind11 fails to compile if we return a
    // std::shared_ptr<std::vector<int64_t>> and include pybind11/stl.h
    HashMap<int64_t, std::vector<int64_t>> map_without_shared_ptr;
    for (const auto& pair : *map_with_shared_ptr) {
      map_without_shared_ptr.emplace(pair.first, *pair.second);
    }
    return map_without_shared_ptr;
  }
};
}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<ParallelDesc, std::shared_ptr<ParallelDesc>>(m, "PlacementSymbol")
      .def(py::init([](int64_t symbol_id, const std::shared_ptr<cfg::ParallelConf>& symbol_conf) {
        return PlacementSymbolExportUtil::ApiCreatePlacementSymbol(symbol_id, symbol_conf);
      }))
      .def(py::init([](const std::string& device_tag,
                       const std::vector<std::string>& machine_device_ids,
                       const std::shared_ptr<Shape>& hierarchy) {
        return PlacementSymbolExportUtil::ApiCreatePlacementSymbol(device_tag, machine_device_ids,
                                                                   hierarchy);
      }))
      .def_property_readonly("symbol_id",
                             [](const ParallelDesc& x) { return x.symbol_id().GetOrThrow(); })
      .def_property_readonly("parallel_conf", &ParallelDesc::cfg_parallel_conf)
      .def_property_readonly("parallel_num", &ParallelDesc::parallel_num)
      .def_property_readonly("device_tag", &ParallelDesc::device_tag)
      .def_property_readonly("machine_id2device_id_list",
                             &PlacementSymbolExportUtil::MachineId2DeviceIdList)
      .def_property_readonly("hierarchy", &ParallelDesc::hierarchy)
      .def("Containing", &ParallelDesc::Bigger)
      .def(py::self == py::self)
      .def(py::hash(py::self));
}

}  // namespace oneflow
