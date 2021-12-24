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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/smart_monkey.h"

namespace oneflow {

namespace {

Maybe<chaos::MonkeyScope> MakeSmartMonkeyScope() {
  OF_CHAOS_MODE_SCOPE(false);
#ifdef OF_ENABLE_CHAOS
  return std::make_shared<chaos::MonkeyScope>(std::make_unique<chaos::SmartMonkey>());
#else
  return Error::CompileOptionWrongError();
#endif
}

}  // namespace

}  // namespace oneflow

ONEFLOW_API_PYBIND11_MODULE("chaos", m) {
  using namespace oneflow;
  namespace py = pybind11;
  py::class_<chaos::MonkeyScope, std::shared_ptr<chaos::MonkeyScope>>(m, "MonkeyGuard")
      .def("failed_cnt", [&](const chaos::MonkeyScope& monkey_scope) {
        return monkey_scope.current_monkey().failed_cnt();
      });
  m.def("smart_monkey_guard", []() { return MakeSmartMonkeyScope().GetPtrOrThrow(); });
}
