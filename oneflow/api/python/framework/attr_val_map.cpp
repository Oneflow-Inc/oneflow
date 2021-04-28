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
#include "oneflow/api/python/of_api_registry.h"

#include "oneflow/core/framework/attr_value_map.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<MutableCfgAttrValueMap, std::shared_ptr<MutableCfgAttrValueMap>>(
      m, "MutableCfgAttrValueMap")
      .def(py::init<>())
      .def("__setitem__",
           [](MutableCfgAttrValueMap* m, const std::string& attr_name,
              const std::shared_ptr<cfg::AttrValue>& attr_value) {
             m->SetAttr(attr_name, attr_value).GetOrThrow();
           })
      .def("__getitem__", [](const MutableCfgAttrValueMap& m,
                             const std::string& attr_name) { return m.at(attr_name); })
      .def(
          "__iter__",
          [](const MutableCfgAttrValueMap& m) { return py::make_iterator(m.begin(), m.end()); },
          py::keep_alive<0, 1>())
      .def("__len__", [](const MutableCfgAttrValueMap& m) { return m.size(); });
}

}  // namespace oneflow
