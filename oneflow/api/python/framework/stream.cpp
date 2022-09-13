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
#include "oneflow/api/python/framework/thread.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/stream_set.h"
#include "oneflow/core/framework/stream_guard.h"

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;
  py::class_<StreamSet, std::shared_ptr<StreamSet>>(m, "StreamSet")
      .def(py::init([](const AsyncThread& async_thread) {
        return StreamSet::New(async_thread.thread_uid()).GetPtrOrThrow();
      }));

  py::class_<StreamGuard, std::shared_ptr<StreamGuard>>(m, "StreamGuard")
      .def(py::init([](const std::shared_ptr<StreamSet>& stream_set) {
        auto stream_converter = std::make_shared<StreamConverter>(stream_set);
        return std::make_shared<StreamGuard>(stream_converter);
      }));
}
