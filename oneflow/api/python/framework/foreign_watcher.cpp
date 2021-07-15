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
#include <string>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/foreign_watcher.h"

namespace py = pybind11;

namespace oneflow {

class PyForeignWatcher : public ForeignWatcher {
 public:
  using ForeignWatcher::ForeignWatcher;

  void Call(const std::string& handler_uuid, int64_t ofblob_ptr) const override {
    PYBIND11_OVERRIDE(void, ForeignWatcher, Call, handler_uuid, ofblob_ptr);
  }
};

}  // namespace oneflow

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;

  py::class_<ForeignWatcher, PyForeignWatcher, std::shared_ptr<ForeignWatcher>>(m, "ForeignWatcher")
      .def(py::init<>())
      .def("Call", &ForeignWatcher::Call);
}
