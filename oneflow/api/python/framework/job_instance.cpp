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
#include "oneflow/core/job/job_instance.h"

namespace py = pybind11;

namespace oneflow {

class PyJobInstance : public JobInstance {
 public:
  // Inherit the constructors
  using JobInstance::JobInstance;

  // Trampoline (need one for each virtual function)
  std::string job_name() const override {
    PYBIND11_OVERRIDE(std::string, /* Return type */
                      JobInstance, /* Parent class */
                      job_name,    /* Name of function in C++ (must match Python name) */
    );
  }

  std::string sole_input_op_name_in_user_job() const override {
    PYBIND11_OVERRIDE(std::string, JobInstance, sole_input_op_name_in_user_job, );
  }

  std::string sole_output_op_name_in_user_job() const override {
    PYBIND11_OVERRIDE(std::string, JobInstance, sole_output_op_name_in_user_job, );
  }

  void PushBlob(uint64_t ofblob_ptr) const override {
    PYBIND11_OVERRIDE(void, JobInstance, PushBlob, ofblob_ptr);
  }

  void PullBlob(uint64_t ofblob_ptr) const override {
    PYBIND11_OVERRIDE(void, JobInstance, PullBlob, ofblob_ptr);
  }

  void Finish() const override { PYBIND11_OVERRIDE(void, JobInstance, Finish, ); }
};

}  // namespace oneflow

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;

  py::class_<JobInstance, PyJobInstance, std::shared_ptr<JobInstance>>(m, "JobInstance")
      .def(py::init<>())
      .def("job_name", &JobInstance::job_name)
      .def("sole_input_op_name_in_user_job", &JobInstance::sole_input_op_name_in_user_job)
      .def("sole_output_op_name_in_user_job", &JobInstance::sole_output_op_name_in_user_job)
      .def("PushBlob", &JobInstance::PushBlob)
      .def("PullBlob", &JobInstance::PullBlob)
      .def("Finish", &JobInstance::Finish);
}
