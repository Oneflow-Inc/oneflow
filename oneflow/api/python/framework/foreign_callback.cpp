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
#include "oneflow/core/job/foreign_callback.h"

namespace py = pybind11;

namespace oneflow {

class PyForeignCallback : public ForeignCallback {
 public:
  // Inherit the constructors
  using ForeignCallback::ForeignCallback;

  // Trampoline (need one for each virtual function)
  void OfBlobCall(int64_t unique_id, int64_t ofblob_ptr) const override {
    PYBIND11_OVERRIDE(void,                 /* Return type */
                      ForeignCallback,      /* Parent class */
                      OfBlobCall,           /* Name of function in C++ (must match Python name) */
                      unique_id, ofblob_ptr /* Argument(s) */
    );
  }

  void RemoveForeignCallback(int64_t unique_id) const override {
    PYBIND11_OVERRIDE(void, ForeignCallback, RemoveForeignCallback, unique_id);
  }
};

}  // namespace oneflow

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;

  py::class_<ForeignCallback, PyForeignCallback, std::shared_ptr<ForeignCallback>>(
      m, "ForeignCallback")
      .def(py::init<>())
      .def("OfBlobCall", &ForeignCallback::OfBlobCall)
      .def("RemoveForeignCallback", &ForeignCallback::RemoveForeignCallback);
}
