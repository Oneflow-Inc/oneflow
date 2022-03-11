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

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("test_api", m) {
  m.def("increase_if_not_none",
        [](const Optional<int>& x) -> Optional<int> { return x.map([](int i) { return i + 1; }); });
  m.def("divide", [](float x, float y) -> Maybe<float> {
    CHECK_NE_OR_RETURN(y, 0);
    return x / y;
  });
  m.def("throw_if_zero", [](int x) -> Maybe<void> {
    CHECK_NE_OR_RETURN(x, 0);
    return Maybe<void>::Ok();
  });
}

}  // namespace oneflow
