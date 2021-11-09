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

#ifdef WITH_MLIR
#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/ir/include/OneFlow/Extension.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/jit_op_interpreter.h"

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("ir", m) {
  m.def("load_jit_shared_lib",
        [](const std::string& lib_path) { MutSharedLibPaths()->insert(lib_path); });
  m.def("toggle_jit", [](const std::string& func_name) {
    *one::MutJitEnabled() = !*one::MutJitEnabled();
    *one::MutJitFuncName() = func_name;
    // when false => true, start jit
    auto jit_interpreter = dynamic_cast<one::JitInterpreter*>(one::GetJitInterpreter().get());
    if (one::IsJitEnabled() == true) { jit_interpreter->Start(); }
    // when true => false, start exec
    if (one::IsJitEnabled() == false) {
      jit_interpreter->Interrupt();
      jit_interpreter->End();
      LOG(ERROR) << "MLIR trace overhead: " << jit_interpreter->MlirTraceOverhead();
    }
    return *one::MutJitEnabled();
  });
  m.def("set_jit_forward_args", [](const std::vector<std::shared_ptr<one::Tensor>>& tensors,
                                   const std::vector<std::shared_ptr<one::Tensor>>& parameters) {
    auto arg_tensors(tensors);
    for (const auto& p : parameters) { arg_tensors.push_back((p)); }
    SetJitForwardArgs(arg_tensors);
  });
}

}  // namespace oneflow
#endif  // WITH_MLIR
