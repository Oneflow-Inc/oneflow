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
#include "oneflow/core/common/foreign_lock_helper.h"

#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/singleton.h"

namespace py = pybind11;

namespace oneflow {
class GILForeignLockHelper final : public ForeignLockHelper {
  Maybe<void> WithScopedRelease(const std::function<Maybe<void>()>& Callback) const override {
    if (PyGILState_Check()) {
      py::gil_scoped_release release;
      JUST(Callback());
    } else {
      JUST(Callback());
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> WithScopedAcquire(const std::function<Maybe<void>()>& Callback) const override {
    if (!PyGILState_Check()) {
      py::gil_scoped_acquire acquire;
      JUST(Callback());
    } else {
      JUST(Callback());
    }
    return Maybe<void>::Ok();
  }
};

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("RegisterGILForeignLockHelper", []() {
    Singleton<ForeignLockHelper>::Delete();
    Singleton<ForeignLockHelper>::SetAllocated(new GILForeignLockHelper());
  });
}

}  // namespace oneflow
