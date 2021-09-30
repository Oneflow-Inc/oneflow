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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/multiprocessing/object_ptr.h"
#include <csignal>

#include <stdexcept>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#define SYSASSERT(rv, ...) \
  if ((rv) < 0) { throw std::system_error(errno, std::system_category(), ##__VA_ARGS__); }

namespace oneflow {
namespace multiprocessing {

namespace py = pybind11;

void multiprocessing_init() {
  auto multiprocessing_module = OFObjectPtr(PyImport_ImportModule("oneflow.multiprocessing"));
  if (!multiprocessing_module) {
    throw std::runtime_error("multiprocessing init error >> multiprocessing_module init fail!");
  }

  auto module = py::handle(multiprocessing_module).cast<py::module>();

  module.def("_prctl_pr_set_pdeathsig", [](int signal) {
#if defined(__linux__)
    auto rv = prctl(PR_SET_PDEATHSIG, signal);
    SYSASSERT(rv, "prctl");
#endif
  });

  // Py_RETURN_TRUE;
}

ONEFLOW_API_PYBIND11_MODULE("", m) { m.def("_multiprocessing_init", &multiprocessing_init); }

}  // namespace multiprocessing
}  // namespace oneflow
