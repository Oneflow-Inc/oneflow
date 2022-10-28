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

#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/autocast.h"

namespace py = pybind11;

namespace oneflow {

class AutoCastMode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoCastMode);

  AutoCastMode(const std::string& device_type, Symbol<DType> dtype, bool enabled,
               bool cache_enabled)
      : prev_enabled_(autocast::is_enabled()),
        prev_cache_enabled_(autocast::is_autocast_cache_enabled()),
        prev_device_type_(autocast::get_autocast_device_type()),
        prev_dtype_(autocast::get_autocast_dtype()),
        prev_gpu_dtype_(autocast::get_autocast_gpu_dtype()),
        prev_cpu_dtype_(autocast::get_autocast_cpu_dtype()) {
    // update autocast state
    autocast::set_enabled(enabled);
    autocast::set_autocast_cache_enabled(cache_enabled);
    if (device_type == "cpu") {
      autocast::set_autocast_device_type(kCPU);
      autocast::set_autocast_dtype(dtype);
      autocast::set_autocast_cpu_dtype(dtype);
    } else if (device_type == "cuda") {
      autocast::set_autocast_device_type(kCUDA);
      autocast::set_autocast_dtype(dtype);
      autocast::set_autocast_gpu_dtype(dtype);
    } else {
      THROW(RuntimeError) << "User specified autocast device_type must be 'cuda' or 'cpu'";
    }
  }

  ~AutoCastMode() {
    autocast::set_enabled(prev_enabled_);
    autocast::set_autocast_cache_enabled(prev_cache_enabled_);
    autocast::set_autocast_device_type(prev_device_type_);
    autocast::set_autocast_dtype(prev_dtype_);
    autocast::set_autocast_gpu_dtype(prev_gpu_dtype_);
    autocast::set_autocast_cpu_dtype(prev_cpu_dtype_);
  }

 private:
  bool prev_enabled_;
  bool prev_cache_enabled_;
  DeviceType prev_device_type_;
  Symbol<DType> prev_dtype_;
  Symbol<DType> prev_gpu_dtype_;
  Symbol<DType> prev_cpu_dtype_;
};

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<AutoCastMode, std::shared_ptr<AutoCastMode>>(m, "AutoCastMode")
      .def(py::init([](const std::string& device_type, Symbol<DType> dtype, bool enabled,
                       bool cache_enabled) {
        return std::make_shared<AutoCastMode>(device_type, dtype, enabled, cache_enabled);
      }));

  m.def("is_autocast_enabled", autocast::is_enabled);
  m.def("set_autocast_enabled", autocast::set_enabled);
  m.def("get_autocast_gpu_dtype", autocast::get_autocast_gpu_dtype);
  m.def("get_autocast_cpu_dtype", autocast::get_autocast_cpu_dtype);
  m.def("set_autocast_gpu_dtype", autocast::set_autocast_gpu_dtype);
  m.def("set_autocast_cpu_dtype", autocast::set_autocast_cpu_dtype);
  m.def("is_autocast_cache_enabled", autocast::is_autocast_cache_enabled);
  m.def("set_autocast_cache_enabled", autocast::set_autocast_cache_enabled);
  m.def("clear_autocast_cache", autocast::clear_cache);
}

}  // namespace oneflow
