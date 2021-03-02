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
#include <vector>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/cfg/pybind_module_registry.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/cfg/message.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<int64_t>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<int64_t, std::shared_ptr<std::vector<int64_t>>>);

namespace oneflow {

namespace {

using IntList = std::vector<int64_t>;
using Int2IntListMap = std::unordered_map<int64_t, std::shared_ptr<IntList>>;

bool Int2IntListMapContaining(const Int2IntListMap& bigger, const Int2IntListMap& smaller) {
  for (const auto& pair : smaller) {
    if (bigger.find(pair.first) == bigger.end()) { return false; }
    const auto& bigger_device_ids = bigger.find(pair.first)->second;
    std::vector<int64_t>::iterator ret;
    for (int64_t device_id : *pair.second) {
      ret = std::find(bigger_device_ids->begin(), bigger_device_ids->end(), device_id);
      if (ret == bigger_device_ids->end()) { return false; }
    }
  }
  return true;
}

}  // namespace

PYBIND11_MODULE(oneflow_api, m) {
  m.def("MasterSendAbort", []() {
    if (Global<EnvGlobalObjectsScope>::Get() != nullptr) {
      return ClusterInstruction::MasterSendAbort();
    }
  });

  using IntList = std::vector<int64_t>;
  using Int2IntListMap = std::unordered_map<int64_t, std::shared_ptr<IntList>>;

  py::module_ oneflow_api_util = m.def_submodule("util");

  py::class_<IntList, std::shared_ptr<IntList>>(oneflow_api_util, "IntList")
      .def(py::init<>())
      .def("__len__", [](const std::shared_ptr<IntList>& v) { return v->size(); })
      .def("items",
           [](std::shared_ptr<IntList>& v) { return py::make_iterator(v->begin(), v->end()); },
           py::keep_alive<0, 1>())
      .def("__getitem__", (IntList::reference & (IntList::*)(IntList::size_type pos)) & IntList::at)
      .def("__iter__",
           [](std::shared_ptr<IntList>& v) { return py::make_iterator(v->begin(), v->end()); },
           py::keep_alive<0, 1>())
      .def("__eq__", [](std::shared_ptr<IntList>& lhs, std::shared_ptr<IntList>& rhs) {
        return *lhs == *rhs;
      });

  py::class_<Int2IntListMap, std::shared_ptr<Int2IntListMap>>(oneflow_api_util, "Int2IntListMap")
      .def(py::init<>())
      .def("__len__", [](const std::shared_ptr<Int2IntListMap>& v) { return v->size(); })
      .def("items",
           [](std::shared_ptr<Int2IntListMap>& v) {
             return py::make_iterator(v->begin(), v->end());
           },
           py::keep_alive<0, 1>())
      .def("__getitem__",
           (Int2IntListMap::mapped_type & (Int2IntListMap::*)(const Int2IntListMap::key_type& pos))
               & Int2IntListMap::operator[])
      .def("__iter__",
           [](std::shared_ptr<Int2IntListMap>& v) {
             return py::make_iterator(v->begin(), v->end());
           },
           py::keep_alive<0, 1>())
      .def("__eq__",
           [](std::shared_ptr<Int2IntListMap>& lhs, std::shared_ptr<Int2IntListMap>& rhs) {
             return Int2IntListMapContaining(*lhs, *rhs) && Int2IntListMapContaining(*rhs, *lhs);
           });

  py::class_<::oneflow::cfg::Message, std::shared_ptr<::oneflow::cfg::Message>>(m, "CfgMessage");
  ::oneflow::cfg::Pybind11ModuleRegistry().ImportAll(m);
  ::oneflow::OneflowModuleRegistry().ImportAll(m);
}

}  // namespace oneflow
