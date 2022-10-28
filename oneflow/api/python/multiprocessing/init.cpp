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
#include "oneflow/core/ep/cpu/cpu_device_manager.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/ep/cpu/cpu_device.h"
#include <csignal>

#include <stdexcept>

#if defined(__linux__)
#include <sys/prctl.h>
#include <system_error>
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

void set_num_threads(int num) {
  int64_t cpu_logic_core = std::thread::hardware_concurrency();
  if (num <= 0) {
    py::print("Warning : ", num, " less than 1 will be set to 1.");
    num = 1;
  } else if (num >= cpu_logic_core) {
    py::print("Warning : ", num,
              " is greater than the number of logical cores and will be set to the maximum number "
              "of logical cores ",
              cpu_logic_core);
    num = cpu_logic_core;
  }

  auto cpu_device = std::static_pointer_cast<ep::CpuDevice>(
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCPU, 0));
  cpu_device->SetNumThreads(num);
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::options options;
  options.disable_function_signatures();
  m.def("_multiprocessing_init", &multiprocessing_init);
  m.def("_set_num_threads", &set_num_threads);
  options.disable_function_signatures();
}

}  // namespace multiprocessing
}  // namespace oneflow
