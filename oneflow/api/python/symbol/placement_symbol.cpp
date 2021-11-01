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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<Shape> MakeShape(const py::tuple& py_shape) {
  DimVector shape_dims{};
  for (const auto& dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
  return std::make_shared<Shape>(shape_dims);
}

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
    JUST(LogicalRun([&parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
      parallel_desc = JUST(builder->GetParallelDescSymbol(parallel_conf));
      return Maybe<void>::Ok();
    }));
    return parallel_desc;
  }

  static std::shared_ptr<ParallelDesc> ApiCreatePlacementSymbol(
      const std::string& device_tag, const std::vector<std::string>& machine_device_ids,
      const std::shared_ptr<Shape>& hierarchy) {
    return CreatePlacementSymbol(device_tag, machine_device_ids, hierarchy).GetPtrOrThrow();
  }

  static Maybe<Symbol<ParallelDesc>> CreatePlacementSymbol(
      const std::string& device_type, const py::dict& machine_device_ids,
      const std::shared_ptr<Shape>& hierarchy) {
    static const HashMap<std::string, std::string> type2device_tag{{"cpu", "cpu"}, {"cuda", "gpu"}};
    CHECK_OR_RETURN(type2device_tag.find(device_type) != type2device_tag.end())
        << "Invalid device_type: " << device_type << ", device_type must be \"cpu\" or \"cuda\".";
    std::string device_tag = type2device_tag.at(device_type);
    std::vector<std::string> formated_machine_device_ids;
    for (const auto& pair : machine_device_ids) {
      CHECK_OR_RETURN(py::isinstance<py::int_>(pair.first))
          << "Key of machine_device_ids must be int.";
      std::string device_name = "";
      std::string machine_id = std::to_string(pair.first.cast<int64_t>());
      if (py::isinstance<py::int_>(pair.second)) {
        device_name = machine_id + ":" + std::to_string(pair.second.cast<int64_t>());
        formated_machine_device_ids.emplace_back(device_name);
      } else {
        CHECK_OR_RETURN(py::isinstance<py::iterable>(pair.second))
            << "Value of machine_device_ids must be int, list or range";
        for (const auto& device_id : pair.second) {
          CHECK_OR_RETURN(py::isinstance<py::int_>(device_id))
              << "Value of machine_device_ids must be int, list or range of int.";
          device_name = machine_id + ":" + std::to_string(device_id.cast<int64_t>());
          formated_machine_device_ids.emplace_back(device_name);
        }
      }
    }
    const auto parallel_conf =
        MakeParallelConf(device_tag, formated_machine_device_ids, hierarchy).GetPtrOrThrow();
    std::shared_ptr<ParallelDesc> parallel_desc;
    JUST(LogicalRun([&parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
      parallel_desc = JUST(builder->GetParallelDescSymbol(parallel_conf));
      return Maybe<void>::Ok();
    }));
    return SymbolOf(*parallel_desc);
  }

  static Symbol<ParallelDesc> ApiCreatePlacementSymbol(const std::string& device_type,
                                                       const py::dict& machine_device_ids,
                                                       const std::shared_ptr<Shape>& hierarchy) {
    return CreatePlacementSymbol(device_type, machine_device_ids, hierarchy).GetOrThrow();
  }

  static Maybe<Symbol<ParallelDesc>> CreatePlacementSymbol(const std::string& device_tag,
                                                           const py::dict& machine_device_ids,
                                                           const py::tuple& hierarchy) {
    std::shared_ptr<Shape> shape = CHECK_JUST(MakeShape(hierarchy));
    return CreatePlacementSymbol(device_tag, machine_device_ids, shape);
  }

  static Symbol<ParallelDesc> ApiCreatePlacementSymbol(const std::string& device_type,
                                                       const py::dict& machine_device_ids,
                                                       const py::tuple& hierarchy) {
    return CreatePlacementSymbol(device_type, machine_device_ids, hierarchy).GetOrThrow();
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

  static Symbol<ParallelDesc> AllDevicePlacement(const std::string& device_type) {
    CHECK_NOTNULL((Global<ResourceDesc, ForEnv>::Get()));
    static const HashMap<std::string, std::string> type2device_tag{{"cpu", "cpu"}, {"cuda", "gpu"}};
    CHECK(type2device_tag.find(device_type) != type2device_tag.end())
        << "Invalid device_type: " << device_type << ", device_type must be \"cpu\" or \"cuda\".";
    std::string device_tag = type2device_tag.at(device_type);
    static thread_local HashMap<std::string, Symbol<ParallelDesc>> device_tag2placement;
    auto iter = device_tag2placement.find(device_tag);
    if (iter == device_tag2placement.end()) {
      int64_t node_size = GlobalProcessCtx::NodeSize();
      int64_t device_num = GlobalProcessCtx::NumOfProcessPerNode();
      std::vector<std::string> machine_device_ids;
      for (int64_t node_id = 0; node_id < node_size; ++node_id) {
        std::string device_name = std::to_string(node_id) + ":0-" + std::to_string(device_num - 1);
        machine_device_ids.emplace_back(device_name);
      }
      Symbol<ParallelDesc> placement =
          SymbolOf(*PlacementSymbolExportUtil::ApiCreatePlacementSymbol(
              device_tag, machine_device_ids, std::shared_ptr<Shape>()));
      iter = device_tag2placement.emplace(device_tag, placement).first;
    }
    return iter->second;
  }

  static std::string PlacementSymbol2String(Symbol<ParallelDesc> placement) {
    return *PlacementToString(placement).GetPtrOrThrow();
  }

  static Maybe<Symbol<ParallelDesc>> ReplacePlacementDeviceTag(Symbol<ParallelDesc> parallel_desc,
                                                               const std::string& device_type) {
    static const HashMap<std::string, std::string> type2device_tag{{"cpu", "cpu"}, {"cuda", "gpu"}};
    std::shared_ptr<cfg::ParallelConf> parallel_conf =
        std::make_shared<cfg::ParallelConf>(*parallel_desc->cfg_parallel_conf());
    parallel_conf->set_device_tag(JUST(MapAt(type2device_tag, device_type)));
    std::shared_ptr<ParallelDesc> out_parallel_desc;
    JUST(LogicalRun(
        [&out_parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
          out_parallel_desc = JUST(builder->GetParallelDescSymbol(parallel_conf));
          return Maybe<void>::Ok();
        }));
    return SymbolOf(*out_parallel_desc);
  }

  static Symbol<ParallelDesc> ApiReplacePlacementDeviceTag(Symbol<ParallelDesc> parallel_desc,
                                                           const std::string& device_type) {
    return ReplacePlacementDeviceTag(parallel_desc, device_type).GetOrThrow();
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

  py::class_<Symbol<ParallelDesc>, std::shared_ptr<Symbol<ParallelDesc>>>(m, "placement",
                                                                          py::dynamic_attr())
      .def(py::init([](const std::string& device_type, const py::dict& machine_device_ids,
                       const std::shared_ptr<Shape>& hierarchy) {
             return PlacementSymbolExportUtil::ApiCreatePlacementSymbol(
                 device_type, machine_device_ids, hierarchy);
           }),
           py::arg("device_type"), py::arg("machine_device_ids"), py::arg("hierarchy"))
      .def(py::init([](const std::string& device_type, const py::dict& machine_device_ids,
                       const py::tuple& hierarchy) {
             return PlacementSymbolExportUtil::ApiCreatePlacementSymbol(
                 device_type, machine_device_ids, hierarchy);
           }),
           py::arg("device_type"), py::arg("machine_device_ids"),
           py::arg("hierarchy") = py::tuple())
      .def_property_readonly("device_type",
                             [](Symbol<ParallelDesc> p) {
                               std::string device_type = p->device_tag() == "gpu" ? "cuda" : "cpu";
                               return device_type;
                             })
      .def_property_readonly("hierarchy", [](Symbol<ParallelDesc> p) { return p->hierarchy(); })
      .def("__str__", &PlacementSymbolExportUtil::PlacementSymbol2String)
      .def("__repr__", &PlacementSymbolExportUtil::PlacementSymbol2String)
      .def(py::self == py::self)
      .def(py::hash(py::self));
  m.def("AllDevicePlacement", &PlacementSymbolExportUtil::AllDevicePlacement);
  m.def("_ReplacePlacementDeviceTag", &PlacementSymbolExportUtil::ApiReplacePlacementDeviceTag);
}

}  // namespace oneflow
