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
#include "oneflow/core/job/foreign_callback_mgr.h"

namespace py = pybind11;

namespace oneflow {

class PyForeignCallback : public ForeignCallback {
 public:
  // Inherit the constructors
  using ForeignCallback::ForeignCallback;

  // Trampoline (need one for each virtual function)
  void EagerMirroredCast(const std::shared_ptr<cfg::OpAttribute>& op_attribute,
                         const std::shared_ptr<cfg::ParallelConf>& parallel_conf) const override {
    PYBIND11_OVERRIDE(void,              /* Return type */
                      ForeignCallback,   /* Parent class */
                      EagerMirroredCast, /* Name of function in C++ (must match Python name) */
                      op_attribute, parallel_conf /* Argument(s) */
    );
  }

  void EagerInterpretCompletedOp(
      const std::shared_ptr<cfg::OpAttribute>& op_attribute,
      const std::shared_ptr<cfg::ParallelConf>& parallel_conf) const override {
    PYBIND11_OVERRIDE(void, ForeignCallback, EagerInterpretCompletedOp, op_attribute,
                      parallel_conf);
  }

  void OfBlobCall(int64_t unique_id, int64_t ofblob_ptr) const override {
    PYBIND11_OVERRIDE(void, ForeignCallback, OfBlobCall, unique_id, ofblob_ptr);
  }

  void RemoveForeignCallback(int64_t unique_id) const override {
    PYBIND11_OVERRIDE(void, ForeignCallback, RemoveForeignCallback, unique_id);
  }

  int64_t MakeScopeSymbol(const std::shared_ptr<cfg::JobConfigProto>& job_conf,
                          const std::shared_ptr<cfg::ParallelConf>& parallel_conf,
                          bool is_mirrored) const override {
    PYBIND11_OVERRIDE(int64_t, ForeignCallback, MakeScopeSymbol, job_conf, parallel_conf,
                      is_mirrored);
  }

  // TODO(lixinqi): remove this urgly api after python code migrated into cpp code
  void AddScopeToPyStorage(int64_t scope_symbol_id,
                           const std::string& scope_proto_str) const override {
    PYBIND11_OVERRIDE(void, ForeignCallback, AddScopeToPyStorage, scope_symbol_id, scope_proto_str);
  }

  int64_t MakeParallelDescSymbol(
      const std::shared_ptr<cfg::ParallelConf>& parallel_conf) const override {
    PYBIND11_OVERRIDE(int64_t, ForeignCallback, MakeParallelDescSymbol, parallel_conf);
  }
};

}  // namespace oneflow

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;

  py::class_<ForeignCallback, PyForeignCallback>(m, "ForeignCallback")
      .def(py::init<>())
      .def("EagerMirroredCast", &ForeignCallback::EagerMirroredCast)
      .def("EagerInterpretCompletedOp", &ForeignCallback::EagerInterpretCompletedOp)
      .def("OfBlobCall", &ForeignCallback::OfBlobCall)
      .def("RemoveForeignCallback", &ForeignCallback::RemoveForeignCallback)
      .def("MakeScopeSymbol", &ForeignCallback::MakeScopeSymbol)
      .def("MakeParallelDescSymbol", &ForeignCallback::MakeParallelDescSymbol);
  m.def("RegisterForeignCallbackOnlyOnce", &RegisterForeignCallbackOnlyOnce);
}
