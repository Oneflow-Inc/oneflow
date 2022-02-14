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

#include "oneflow/extension/python/numpy.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif  // WITH_CUDA

namespace py = pybind11;

namespace oneflow {

namespace {

int64_t GetGpuDeviceNum() {
#ifndef WITH_CUDA
  return 0;
#else
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  return device_count;
#endif
}

struct PlacementSymbolExportUtil {
  static Maybe<std::string> GetDeviceTag(const std::string& type) {
    static const HashMap<std::string, std::string> type2device_tag{{"cpu", "cpu"}, {"cuda", "gpu"}};
    const auto& it = type2device_tag.find(type);
    if (it == type2device_tag.end()) {
      return Error::RuntimeError() << "placement type should only be cpu or cuda, but got " << type;
    }
    return it->second;
  }

  static Maybe<ParallelDesc> CreateParallelDesc(
      const std::string& type, const std::vector<std::string>& formated_machine_device_ids,
      const std::shared_ptr<Shape>& hierarchy_shape) {
    CHECK_OR_RETURN(type == "cpu" || type == "cuda")
        << "placement type must be \"cpu\" or \"cuda\".";
    const auto& device_tag = JUST(GetDeviceTag(type));
    auto parallel_conf =
        JUST(MakeParallelConf(*device_tag, formated_machine_device_ids, hierarchy_shape));
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
        << "placement ranks shoule be array of int64.";
    int64_t* rank_data = static_cast<int64_t*>(PyArray_DATA(ranks));

    std::vector<std::pair<int64_t, int64_t>> machine_device_id_vec;
    for (int i = 0; i < size; ++i) {
      int64_t rank = rank_data[i];
      // TODO(hjchen2): Prevent users from creating illegal placement
      // if (rank >= GlobalProcessCtx::WorldSize()) {
      //   return Error::RuntimeError() << "rank " << rank << " is invalid since the world size is "
      //                                << GlobalProcessCtx::WorldSize();
      // }
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
    if (!obj) { return Error::RuntimeError() << "placement ranks must be int64 array."; }

    const auto& shape = JUST(GetRanksShape(obj));
    const auto& formated_machine_device_ids = JUST(ParseAndFormatRanks(obj));
    return SymbolOf(*JUST(CreateParallelDesc(type, *formated_machine_device_ids, shape)));
  }

  static Maybe<Symbol<ParallelDesc>> AllDevicePlacement(const std::string& type) {
    static thread_local HashMap<std::string, Symbol<ParallelDesc>> device_tag2placement;
    CHECK_NOTNULL((Global<ResourceDesc, ForEnv>::Get()));
    const auto& device_tag = JUST(GetDeviceTag(type));
    auto it = device_tag2placement.find(*device_tag);
    if (it == device_tag2placement.end()) {
      int64_t node_size = GlobalProcessCtx::NodeSize();
      int64_t device_num = GlobalProcessCtx::NumOfProcessPerNode();
      if (*device_tag == "gpu") {
        const int64_t gpu_device_num = GetGpuDeviceNum();
        CHECK_NE_OR_RETURN(gpu_device_num, 0)
            << "Can\'t construct placment with \"cuda\" type because there is no CUDA device!";
        device_num = std::min(device_num, gpu_device_num);
      }
      std::vector<std::string> machine_device_ids;
      for (int64_t node_id = 0; node_id < node_size; ++node_id) {
        std::string device_name = std::to_string(node_id) + ":0-" + std::to_string(device_num - 1);
        machine_device_ids.emplace_back(device_name);
      }
      Symbol<ParallelDesc> placement =
          SymbolOf(*JUST(CreateParallelDesc(type, machine_device_ids, std::shared_ptr<Shape>())));
      it = device_tag2placement.emplace(*device_tag, placement).first;
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
      .def_property_readonly(
          "device_type",
          [](Symbol<ParallelDesc> p) {
            PyErr_WarnEx(
                PyExc_UserWarning,
                "The property .device_type of placement is deprecated, please use .type instead",
                1);
            return p->device_tag() == "gpu" ? "cuda" : "cpu";
          })
      .def_property_readonly(
          "type", [](Symbol<ParallelDesc> p) { return p->device_tag() == "gpu" ? "cuda" : "cpu"; })
      .def_property_readonly("hierarchy",
                             [](Symbol<ParallelDesc> p) {
                               PyErr_WarnEx(PyExc_UserWarning,
                                            "The property .hierarchy of placement is deprecated, "
                                            "please use .ranks.shape instead",
                                            1);
                               return p->hierarchy();
                             })
      .def_property_readonly("ranks",
                             [](Symbol<ParallelDesc> p) {
                               return PlacementSymbolExportUtil::GetPlacementRanks(p).GetOrThrow();
                             })
      .def("__str__", [](Symbol<ParallelDesc> p) { return PlacementToString(p).GetOrThrow(); })
      .def("__repr__", [](Symbol<ParallelDesc> p) { return PlacementToString(p).GetOrThrow(); })
      .def(py::self == py::self)
      .def(py::hash(py::self));
  m.def("AllDevicePlacement", [](const std::string& type) {
    return PlacementSymbolExportUtil::AllDevicePlacement(type).GetOrThrow();
  });
}

}  // namespace oneflow
