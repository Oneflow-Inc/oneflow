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
#include <algorithm>
#include <cstdint>
#include <memory>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "oneflow/core/common/maybe.h"
#include "oneflow/extension/python/numpy.h"
#include "oneflow/core/common/error.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/framework/placement_utils.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif  // WITH_CUDA

namespace py = pybind11;

namespace oneflow {

namespace {

class Py11Placement {
 public:
  Py11Placement(const std::string& device_type)
      : device_type_(device_type), device_tag_(device_type) {
    CHECK(device_type_ == "cpu" || device_type_ == "cuda") << "Only support cpu or cuda device!";
    if (device_type_ == "cuda") device_tag_ = "gpu";
    node_size_ = GlobalProcessCtx::NodeSize();
    device_num_ = GetDeviceNum(device_type);
  }

  Symbol<ParallelDesc> ApiCreateParallelDescSymbol(const py::object& ranks) {
    return CreateParallelDescSymbol(ranks).GetOrThrow();
  }

  py::list GetRankList(const Symbol<ParallelDesc>& para_desc) const {
    py::list py11_rank_list;
    auto hierarchy = para_desc->hierarchy();
    auto rank_ids = GetSortedRankIds(para_desc);
    if (hierarchy->NumAxes() == 1) {
      for (auto id : rank_ids) { py11_rank_list.append(id); }
    } else if (hierarchy->NumAxes() == 2) {
      for (auto i = 0; i < hierarchy->At(0); i++) {
        py::list tmp;
        for (auto j = 0; j < hierarchy->At(1); j++) {
          tmp.append(rank_ids[i * hierarchy->At(1) + j]);
        }
        py11_rank_list.append(tmp);
      }
    }
    return py11_rank_list;
  }

  Symbol<ParallelDesc> GetAllDevicePlacement() {
    CHECK_NOTNULL((Global<ResourceDesc, ForEnv>::Get()));
    auto hierarchy_shape = std::make_shared<Shape>();
    std::vector<std::string> machine_device_ids;
    for (auto node_id = 0; node_id < node_size_; node_id++) {
      std::string device_name = std::to_string(node_id) + ":0-" + std::to_string(device_num_ - 1);
      machine_device_ids.emplace_back(device_name);
    }

    auto symbol = CreateParallelDesc(machine_device_ids, hierarchy_shape).GetOrThrow();
    return SymbolOf(symbol);
  }

 private:
  // get all rank ids through sorted_machine_ids interface in ParallelDesc,
  // in the end sort all the rank ids to 0,1,2,3.... and return
  std::vector<int64_t> GetSortedRankIds(const Symbol<ParallelDesc>& para_desc) const {
    std::vector<int64_t> rank_ids;
    for (auto process_id : para_desc->sorted_machine_ids()) {
      int64_t node_id = GlobalProcessCtx::NodeId(process_id);
      for (auto device_id : para_desc->sorted_dev_phy_ids(process_id)) {
        rank_ids.push_back(node_id * device_num_ + device_id);
      }
    }
    std::sort(rank_ids.begin(), rank_ids.end());
    return rank_ids;
  }

  // create Symbol<ParallelDesc> object through given device_type and ranks parameters
  Maybe<Symbol<ParallelDesc>> CreateParallelDescSymbol(const py::object& ranks) {
    auto* obj = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
        ranks.ptr(), nullptr, 0, 0, NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSURECOPY, nullptr));
    if (!obj) { return Error::RuntimeError() << "ranks parameter error!"; }

    auto hierarchy_shape = JUST(GetHierarchyShape(obj));
    auto rank_num = hierarchy_shape->elem_cnt();
    auto* rank_ids = static_cast<int64_t*>(PyArray_DATA(obj));
    JUST(CheckNoRepeat(rank_ids, rank_num));
    auto formated_machine_device_ids = JUST(GetFormatDevInfo(rank_ids, rank_num));

    auto parallel_desc = JUST(CreateParallelDesc(*formated_machine_device_ids, hierarchy_shape));
    return SymbolOf(*parallel_desc);
  }

  Maybe<ParallelDesc> CreateParallelDesc(
      const std::vector<std::string>& formated_machine_device_ids,
      std::shared_ptr<Shape> hierarchy_shape) {
    auto parallel_conf =
        JUST(MakeParallelConf(device_tag_, formated_machine_device_ids, hierarchy_shape));
    std::shared_ptr<ParallelDesc> parallel_desc;
    JUST(PhysicalRun([&parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
      parallel_desc = JUST(builder->GetParallelDescSymbol(parallel_conf));
      return Maybe<void>::Ok();
    }));

    return parallel_desc;
  }

  // parse hierarchy shape out from PyArrayObject
  Maybe<Shape> GetHierarchyShape(PyArrayObject* py_arr) {
    CHECK_OR_RETURN(py_arr != nullptr && PyArray_NDIM(py_arr) <= 2)
        << "ranks parameter error! hierarchy dimension is " << PyArray_NDIM(py_arr);

    auto* dims_ptr = PyArray_SHAPE(py_arr);
    return std::make_shared<Shape>(DimVector(dims_ptr, dims_ptr + PyArray_NDIM(py_arr)));
  }

  // check whether given rank_ids have repeated rank_id value
  Maybe<bool> CheckNoRepeat(const int64_t* rank_ids, int64_t rank_num) {
    std::unordered_set<int64_t> tmp;
    for (auto i = 0; i < rank_num; i++) {
      auto rank_id = rank_ids[i];
      CHECK_OR_RETURN(tmp.count(rank_id) == 0)
          << "ranks parameter error! giving multi rank id " << rank_id;
      tmp.insert(rank_id);
    }
    return true;
  }

  // transform integral rank_ids to following string format:
  // 0:0
  // 0:1
  // 1:0
  // 1:1
  // ...
  Maybe<std::vector<std::string>> GetFormatDevInfo(const int64_t* rank_ids, int64_t rank_num) {
    std::vector<std::pair<int64_t, int64_t>> machine_device_id_vec;
    for (int i = 0; i < rank_num; ++i) {
      auto rank_id = rank_ids[i];
      auto machine_id = rank_id / device_num_;
      CHECK_OR_RETURN(machine_id < node_size_)
          << "Error: node size " << node_size_ << ", device number " << device_num_ << ", rank id "
          << rank_id << ", machine id " << machine_id;
      auto device_id = rank_id % device_num_;
      machine_device_id_vec.emplace_back(machine_id, device_id);
    }

    std::vector<std::string> formated_machine_device_ids;
    for (const auto& pair : machine_device_id_vec) {
      auto device_name = std::to_string(pair.first) + ":" + std::to_string(pair.second);
      formated_machine_device_ids.emplace_back(device_name);
    }
    return formated_machine_device_ids;
  }

  int64_t GetDeviceNum(const std::string& device) const {
    auto device_num = GlobalProcessCtx::NumOfProcessPerNode();
    if (device == "gpu" || device == "cuda") {
      int gpu_device_num = 0;
#ifdef WITH_CUDA
      cudaGetDeviceCount(&gpu_device_num);
#endif
      CHECK_NE(gpu_device_num, 0)
          << "Can\'t construct placment with \"cuda\" type because there is no CUDA device!";
      if (device_num > gpu_device_num) device_num = gpu_device_num;
    }
    return device_num;
  }

 private:
  std::string device_type_;
  std::string device_tag_;
  int64_t node_size_;
  int64_t device_num_;
};

Maybe<Shape> MakeShape(const py::tuple& py_shape) {
  DimVector shape_dims{};
  for (const auto& dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
  return std::make_shared<Shape>(shape_dims);
}

struct PlacementSymbolExportUtil {
  static Maybe<Symbol<ParallelDesc>> CreatePlacementSymbol(
      const std::string& device_type, const py::iterable& device_ids,
      const std::shared_ptr<Shape>& hierarchy) {
    std::vector<std::pair<int64_t, int64_t>> machine_device_id_vec;
    if (py::isinstance<py::dict>(device_ids)) {
      const py::dict& machine_device_id_dict = device_ids.cast<py::dict>();
      for (const auto& pair : machine_device_id_dict) {
        CHECK_OR_RETURN(py::isinstance<py::int_>(pair.first))
            << "Key of device_ids dict must be int.";
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
    } else {
      for (const auto& global_device_id : device_ids) {
        CHECK_OR_RETURN(py::isinstance<py::int_>(global_device_id))
            << "Value of device_ids list must be int";
        int64_t global_rank_int64 = global_device_id.cast<int64_t>();
        machine_device_id_vec.emplace_back(GlobalProcessCtx::NodeId(global_rank_int64),
                                           GlobalProcessCtx::LocalRank(global_rank_int64));
      }
    }

    static const HashMap<std::string, std::string> type2device_tag{{"cpu", "cpu"}, {"cuda", "gpu"}};
    CHECK_OR_RETURN(type2device_tag.find(device_type) != type2device_tag.end())
        << "Invalid device_type: " << device_type << ", device_type must be \"cpu\" or \"cuda\".";
    const std::string& device_tag = type2device_tag.at(device_type);
    std::vector<std::string> formated_machine_device_ids;
    for (const auto& pair : machine_device_id_vec) {
      const std::string& device_name =
          std::to_string(pair.first) + ":" + std::to_string(pair.second);
      formated_machine_device_ids.emplace_back(device_name);
    }
    const auto parallel_conf =
        JUST(MakeParallelConf(device_tag, formated_machine_device_ids, hierarchy));
    std::shared_ptr<ParallelDesc> parallel_desc;
    JUST(PhysicalRun([&parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
      parallel_desc = JUST(builder->GetParallelDescSymbol(parallel_conf));
      return Maybe<void>::Ok();
    }));
    return SymbolOf(*parallel_desc);
  }

  static Symbol<ParallelDesc> ApiCreatePlacementSymbol(const std::string& device_type,
                                                       const py::iterable& device_ids,
                                                       const std::shared_ptr<Shape>& hierarchy) {
    return CreatePlacementSymbol(device_type, device_ids, hierarchy).GetOrThrow();
  }

  static std::string PlacementSymbol2String(Symbol<ParallelDesc> placement) {
    return *PlacementToString(placement).GetPtrOrThrow();
  }
};

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<ParallelDesc>, std::shared_ptr<Symbol<ParallelDesc>>>(m, "placement",
                                                                          py::dynamic_attr())
      .def(py::init([](const std::string& device_type, const py::iterable& device_ids,
                       const py::tuple& hierarchy) {
             std::shared_ptr<Shape> hierarchy_shape = MakeShape(hierarchy).GetPtrOrThrow();
             return PlacementSymbolExportUtil::ApiCreatePlacementSymbol(device_type, device_ids,
                                                                        hierarchy_shape);
           }),
           py::arg("device_type"), py::arg("device_ids"), py::arg("hierarchy") = py::tuple())
      .def(py::init([](const std::string& device_type, const py::object& ranks) {
             auto t = std::make_unique<Py11Placement>(device_type);
             return t->ApiCreateParallelDescSymbol(ranks);
           }),
           py::arg("type"), py::arg("ranks"))
      .def_property_readonly("device_type",
                             [](Symbol<ParallelDesc> p) {
                               std::string device_type = p->device_tag() == "gpu" ? "cuda" : "cpu";
                               return device_type;
                             })
      .def_property_readonly("type",
                             [](Symbol<ParallelDesc> p) {
                               std::string device_type = p->device_tag() == "gpu" ? "cuda" : "cpu";
                               return device_type;
                             })
      .def_property_readonly("ranks",
                             [](Symbol<ParallelDesc> p) {
                               std::string device_type = p->device_tag() == "gpu" ? "cuda" : "cpu";
                               auto t = std::make_unique<Py11Placement>(device_type);
                               return t->GetRankList(p);
                             })
      .def_property_readonly("device_ids",
                             [](Symbol<ParallelDesc> p) {
                               std::map<int64_t, py::list> device_ids;
                               for (int64_t machine_id : p->sorted_machine_ids()) {
                                 int64_t node_id = GlobalProcessCtx::NodeId(machine_id);
                                 for (int64_t device_id : p->sorted_dev_phy_ids(machine_id)) {
                                   device_ids[node_id].append(py::cast(device_id));
                                 }
                               }
                               return device_ids;
                             })
      .def_property_readonly("hierarchy", [](Symbol<ParallelDesc> p) { return p->hierarchy(); })
      .def("__str__", &PlacementSymbolExportUtil::PlacementSymbol2String)
      .def("__repr__", &PlacementSymbolExportUtil::PlacementSymbol2String)
      .def(py::self == py::self)
      .def(py::hash(py::self));
  m.def("AllDevicePlacement", [](const std::string& device_type) {
    auto t = std::make_unique<Py11Placement>(device_type);
    return t->GetAllDevicePlacement();
  });
}

}  // namespace oneflow
