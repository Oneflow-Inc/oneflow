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
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/user_op_attr.pb.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<MutableCfgAttrMap, std::shared_ptr<MutableCfgAttrMap>>(m, "MutableCfgAttrMap")
      .def(py::init<>())
      .def("__setitem__",
           [](MutableCfgAttrMap* m, const std::string& attr_name,
              const std::string& attr_value_str) {
             std::shared_ptr<AttrValue> attr_value = std::make_shared<AttrValue>();
             if (!TxtString2PbMessage(attr_value_str, attr_value.get())) {
               THROW(RuntimeError) << "attr value parse failed.\n" << attr_value_str;
             }
             return m->SetAttr(attr_name, attr_value);
           })
      .def("__getitem__",
           [](const MutableCfgAttrMap& m, const std::string& attr_name) { return m.at(attr_name); })
      .def(
          "__iter__",
          [](const MutableCfgAttrMap& m) { return py::make_iterator(m.begin(), m.end()); },
          py::keep_alive<0, 1>())
      .def("__len__", [](const MutableCfgAttrMap& m) { return m.size(); });
}

}  // namespace oneflow
