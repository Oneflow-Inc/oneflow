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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "oneflow/core/common/maybe.h"
#include "oneflow/extension/python/numpy.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace py = pybind11;

namespace oneflow {

namespace {

int64_t GetDeviceCount(const std::string& device_name) {
  return Singleton<ep::DeviceManagerRegistry>::Get()->GetDeviceCount(device_name);
}

struct PlacementSymbolExportUtil {
  static Maybe<void> CheckDeviceTag(const std::string& type) {
    if (!TRY(DeviceType4DeviceTag(type)).IsOk()) {
      return Error::RuntimeError() << "Expected one of " << PrintAvailableDevices()
                                   << " device type at start of device string: " << type;
    }
    return Maybe<void>::Ok();
  }

  static Maybe<ParallelDesc> CreateParallelDesc(
      const std::string& type, const std::vector<std::string>& formated_machine_device_ids,
      const std::shared_ptr<Shape>& hierarchy_shape) {
    JUST(CheckDeviceTag(type));
    auto parallel_conf = JUST(MakeParallelConf(type, formated_machine_device_ids, hierarchy_shape));
    std::shared_ptr<ParallelDesc> parallel_desc;
    JUST(PhysicalRun([&parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
      parallel_desc = JUST(builder->GetParallelDescSymbol(*parallel_conf));
      return Maybe<void>::Ok();
    }));

    return parallel_desc;
  }

  static Maybe<ParallelDesc> CreateParallelDesc(const std::string& proto_str) {
    ParallelConf parallel_conf;
    CHECK_OR_RETURN(TxtString2PbMessage(proto_str, &parallel_conf))
        << " Get ParallelConf Pb from string failed.";
    std::shared_ptr<ParallelDesc> parallel_desc;
    JUST(PhysicalRun([&parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
      parallel_desc = JUST(builder->GetParallelDescSymbol(parallel_conf));
      return Maybe<void>::Ok();
    }));

    return parallel_desc;
  }

  static Maybe<std::vector<std::string>> ParseAndFormatRanks(const py::dict& device_ids) {
    std::vector<std::pair<int64_t, int64_t>> machine_device_id_vec;
    for (const auto& pair : device_ids) {
      CHECK_OR_RETURN(py::isinstance<py::int_>(pair.first))
          << "The key (node id) of placement device_ids must be int64.";
      int64_t machine_id = pair.first.cast<int64_t>();
      if (py::isinstance<py::int_>(pair.second)) {
        machine_device_id_vec.emplace_back(machine_id, pair.second.cast<int64_t>());
      } else {
        CHECK_OR_RETURN(py::isinstance<py::iterable>(pair.second))
            << "Value of device_ids dict must be int, list or range";
        for (const auto& device_id : pair.second) {
          CHECK_OR_RETURN(py::isinstance<py::int_>(device_id))
              << "Value of device_ids dict must be int, list or range of int.";
          machine_device_id_vec.emplace_back(machine_id, device_id.cast<int64_t>());
        }
      }
    }
    auto formated_machine_device_ids = std::make_shared<std::vector<std::string>>();
    for (const auto& pair : machine_device_id_vec) {
      const std::string& device_name =
          std::to_string(pair.first) + ":" + std::to_string(pair.second);
      formated_machine_device_ids->emplace_back(device_name);
    }
    return formated_machine_device_ids;
  }

  static Maybe<Shape> GetRanksShape(PyArrayObject* ranks) {
    auto* shape = PyArray_SHAPE(ranks);
    return std::make_shared<Shape>(DimVector(shape, shape + PyArray_NDIM(ranks)));
  }

  // Parse and format ranks to string "machine_id:local_rank"
  static Maybe<std::vector<std::string>> ParseAndFormatRanks(PyArrayObject* ranks) {
    size_t size = PyArray_SIZE(ranks);
    CHECK_EQ_OR_RETURN(PyArray_TYPE(ranks), NPY_INT64)
        << Error::RuntimeError() << "placement ranks shoule be an array of long int";
    int64_t* rank_data = static_cast<int64_t*>(PyArray_DATA(ranks));

    std::vector<std::pair<int64_t, int64_t>> machine_device_id_vec;
    for (int i = 0; i < size; ++i) {
      int64_t rank = rank_data[i];
      int64_t machine_id = GlobalProcessCtx::NodeId(rank);
      int64_t device_id = GlobalProcessCtx::LocalRank(rank);
      machine_device_id_vec.emplace_back(machine_id, device_id);
    }

    auto formated_machine_device_ids = std::make_shared<std::vector<std::string>>();
    for (const auto& pair : machine_device_id_vec) {
      auto device_name = std::to_string(pair.first) + ":" + std::to_string(pair.second);
      formated_machine_device_ids->emplace_back(device_name);
    }
    return formated_machine_device_ids;
  }

  static Maybe<Symbol<ParallelDesc>> CreateParallelDescSymbol(
      const std::string& type, const py::dict& device_ids,
      const std::shared_ptr<Shape>& hierarchy) {
    const auto& formated_machine_device_ids = JUST(ParseAndFormatRanks(device_ids));
    return SymbolOf(*JUST(CreateParallelDesc(type, *formated_machine_device_ids, hierarchy)));
  }

  // create Symbol<ParallelDesc> object through given device_type and ranks parameters
  static Maybe<Symbol<ParallelDesc>> CreateParallelDescSymbol(const std::string& type,
                                                              const py::object& ranks) {
    auto* obj = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
        ranks.ptr(), nullptr, 0, 0, NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSURECOPY, nullptr));
    if (!obj) { return Error::RuntimeError() << "placement ranks shoule be an array of long int"; }

    const auto& shape = JUST(GetRanksShape(obj));
    const auto& formated_machine_device_ids = JUST(ParseAndFormatRanks(obj));
    return SymbolOf(*JUST(CreateParallelDesc(type, *formated_machine_device_ids, shape)));
  }

  static Maybe<Symbol<ParallelDesc>> CreateParallelDescSymbol(const std::string& proto_str) {
    return SymbolOf(*JUST(CreateParallelDesc(proto_str)));
  }

  static Maybe<Symbol<ParallelDesc>> AllDevicePlacement(const std::string& type) {
    static thread_local HashMap<std::string, Symbol<ParallelDesc>> device_tag2placement;
    CHECK_NOTNULL((Singleton<ResourceDesc, ForEnv>::Get()));
    JUST(CheckDeviceTag(type));
    auto it = device_tag2placement.find(type);
    if (it == device_tag2placement.end()) {
      int64_t node_size = GlobalProcessCtx::NodeSize();
      int64_t device_num = GlobalProcessCtx::NumOfProcessPerNode();
      if (type != "cpu") {
        const int64_t device_count = GetDeviceCount(type);
        CHECK_NE_OR_RETURN(device_count, 0)
            << Error::RuntimeError() << "Can\'t construct placement with \"" << type
            << "\" type because there is no device!";
        device_num = std::min(device_num, device_count);
      }
      std::vector<std::string> machine_device_ids;
      for (int64_t node_id = 0; node_id < node_size; ++node_id) {
        std::string device_name = std::to_string(node_id) + ":0-" + std::to_string(device_num - 1);
        machine_device_ids.emplace_back(device_name);
      }
      Symbol<ParallelDesc> placement =
          SymbolOf(*JUST(CreateParallelDesc(type, machine_device_ids, std::shared_ptr<Shape>())));
      it = device_tag2placement.emplace(type, placement).first;
    }
    return it->second;
  }

  static Maybe<py::array> GetPlacementRanks(const Symbol<ParallelDesc>& placement) {
    py::list ranks;
    for (int64_t machine_id : placement->sorted_machine_ids()) {
      int64_t node_id = GlobalProcessCtx::NodeId(machine_id);
      for (int64_t device_id : placement->sorted_dev_phy_ids(machine_id)) {
        ranks.append(py::cast(node_id * GlobalProcessCtx::NumOfProcessPerNode() + device_id));
      }
    }
    auto array_ranks = py::cast<py::array>(ranks);
    array_ranks.resize(placement->hierarchy()->dim_vec());
    return array_ranks;
  }
};

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<ParallelDesc>, std::shared_ptr<Symbol<ParallelDesc>>>(m, "placement",
                                                                          py::dynamic_attr())
      .def(py::init([](const std::string& device_type, const py::dict& device_ids,
                       const std::shared_ptr<Shape>& hierarchy) {
             PyErr_WarnEx(
                 PyExc_UserWarning,
                 "The way to construct placement is deprecated, and it will be removed in next "
                 "versions. Please use oneflow.placement(type=str, ranks=int array) instead",
                 1);
             return PlacementSymbolExportUtil::CreateParallelDescSymbol(device_type, device_ids,
                                                                        hierarchy)
                 .GetOrThrow();
           }),
           py::arg("device_type"), py::arg("device_ids"), py::arg("hierarchy"))
      .def(py::init([](const std::string& device_type, const py::dict& device_ids,
                       const py::tuple& hierarchy) {
             PyErr_WarnEx(
                 PyExc_UserWarning,
                 "The way to construct placement is deprecated, and it will be removed in next "
                 "versions. Please use oneflow.placement(type=str, ranks=int array) instead",
                 1);
             DimVector shape_dims{};
             for (const auto& dim : hierarchy) { shape_dims.emplace_back(dim.cast<int64_t>()); }
             return PlacementSymbolExportUtil::CreateParallelDescSymbol(
                        device_type, device_ids, std::make_shared<Shape>(shape_dims))
                 .GetOrThrow();
           }),
           py::arg("device_type"), py::arg("device_ids"), py::arg("hierarchy") = py::tuple())
      .def(py::init([](const std::string& type, const py::object& ranks) {
             return PlacementSymbolExportUtil::CreateParallelDescSymbol(type, ranks).GetOrThrow();
           }),
           py::arg("type"), py::arg("ranks"))
      .def(py::init([](const std::string& proto_str) {
             return PlacementSymbolExportUtil::CreateParallelDescSymbol(proto_str).GetOrThrow();
           }),
           py::arg("proto_str"))
      .def_property_readonly(
          "device_type",
          [](Symbol<ParallelDesc> p) {
            PyErr_WarnEx(
                PyExc_UserWarning,
                "The property .device_type of placement is deprecated, please use .type instead",
                1);
            return p->device_tag();
          })
      .def_property_readonly("type", [](Symbol<ParallelDesc> p) { return p->device_tag(); })
      .def_property_readonly("hierarchy",
                             [](Symbol<ParallelDesc> p) {
                               PyErr_WarnEx(PyExc_UserWarning,
                                            "The property .hierarchy of placement is deprecated, "
                                            "please use .ranks.shape instead",
                                            1);
                               return p->hierarchy();
                             })
      .def_property_readonly("ranks", &PlacementSymbolExportUtil::GetPlacementRanks)
      .def("__str__", PlacementToString)
      .def("__repr__", PlacementToString)
      .def(py::self == py::self)
      .def(py::hash(py::self))
      .def_static("all", &PlacementSymbolExportUtil::AllDevicePlacement);
}

}  // namespace oneflow
