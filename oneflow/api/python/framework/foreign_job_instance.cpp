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
#include <memory>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/foreign_job_instance.h"

namespace py = pybind11;

namespace oneflow {

class PyForeignJobInstance : public ForeignJobInstance {
 public:
  // Inherit the constructors
  using ForeignJobInstance::ForeignJobInstance;

  // Trampoline (need one for each virtual function)
  std::string job_name() const override {
    PYBIND11_OVERRIDE(std::string,        /* Return type */
                      ForeignJobInstance, /* Parent class */
                      job_name,           /* Name of function in C++ (must match Python name) */
    );
  }

  std::string sole_input_op_name_in_user_job() const override {
    PYBIND11_OVERRIDE(std::string, ForeignJobInstance, sole_input_op_name_in_user_job, );
  }

  std::string sole_output_op_name_in_user_job() const override {
    PYBIND11_OVERRIDE(std::string, ForeignJobInstance, sole_output_op_name_in_user_job, );
  }

  void PushBlob(uint64_t ofblob_ptr) const override {
    PYBIND11_OVERRIDE(void, ForeignJobInstance, PushBlob, ofblob_ptr);
  }

  void PullBlob(uint64_t ofblob_ptr) const override {
    PYBIND11_OVERRIDE(void, ForeignJobInstance, PullBlob, ofblob_ptr);
  }

  void Finish() const override { PYBIND11_OVERRIDE(void, ForeignJobInstance, Finish, ); }
};

}  // namespace oneflow

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;

  py::class_<ForeignJobInstance, PyForeignJobInstance, std::shared_ptr<ForeignJobInstance>>(
      m, "ForeignJobInstance")
      .def(py::init<>())
      .def("job_name", &ForeignJobInstance::job_name)
      .def("sole_input_op_name_in_user_job", &ForeignJobInstance::sole_input_op_name_in_user_job)
      .def("sole_output_op_name_in_user_job", &ForeignJobInstance::sole_output_op_name_in_user_job)
      .def("PushBlob", &ForeignJobInstance::PushBlob)
      .def("PullBlob", &ForeignJobInstance::PullBlob)
      .def("Finish", &ForeignJobInstance::Finish);
}
