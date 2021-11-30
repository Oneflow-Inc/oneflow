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

#include <utility>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/ir/include/OneFlow/Extension.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/ir/oneflow-jit/include/OneFlow/jit_op_interpreter.h"
#include "oneflow/api/python/functional/common.h"

namespace py = pybind11;

namespace oneflow {

using namespace one::ir;

namespace jit {
struct Module {
  explicit Module(py::object py_module)
      : py_module_(std::move(py_module)),
        context_(new MLIRContext()),
        module_(CreateJitModule(context_)),
        importer_(context_, *module_) {}
  py::object forward(const py::args& args, const py::kwargs& kwargs) {
    auto jit_interpreter = dynamic_cast<one::JitInterpreter*>(one::GetJitInterpreter().get());
    *one::MutJitEnabled() = true;
    auto i_this = reinterpret_cast<std::uintptr_t>(this);
    std::string func_name = "jitModule" + std::to_string(i_this) + "_" + std::to_string(nth_call_);
    auto inputs = args.cast<std::vector<std::shared_ptr<one::Tensor>>>();
    auto parameters_generator = py_module_.attr("parameters")();
    std::vector<std::shared_ptr<one::Tensor>> parameters{};
    for (auto p : parameters_generator) {
      parameters.push_back(p.cast<std::shared_ptr<one::Tensor>>());
    }
    std::vector<std::shared_ptr<one::Tensor>> arg_tensors{};
    arg_tensors.insert(arg_tensors.end(), inputs.begin(), inputs.end());
    arg_tensors.insert(arg_tensors.end(), parameters.begin(), parameters.end());
    py::object ret;
    std::vector<std::shared_ptr<one::Tensor>> tensors_to_materialize{};
    jit_interpreter->Trace(importer_, func_name, arg_tensors, [&]() {
      ret = py_module_.attr("forward")(*args, **kwargs);
      if (auto tensor = ret.cast<std::shared_ptr<one::Tensor>>()) {
        tensors_to_materialize.push_back(tensor);
      }
      return tensors_to_materialize;
    });
    *one::MutJitEnabled() = false;
    jit_interpreter->DispatchModule(*module_, one::GetJitFuncName(), arg_tensors,
                                    tensors_to_materialize);
    importer_.ResetMappings();
    jit_interpreter->End();
    LOG(ERROR) << "JIT trace overhead: " << jit_interpreter->TraceOverhead();
    nth_call_ += 1;
    return ret;
  }
  void dump_ir() { module_->dump(); }

 private:
  py::object py_module_;
  MLIRContext* context_;
  OwningOpRef<ModuleOp> module_;
  JitImporter importer_;
  int32_t nth_call_ = 0;  // TODO: remove this dirty workaround
};
}  // namespace jit

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
      LOG(ERROR) << "JIT trace overhead: " << jit_interpreter->TraceOverhead();
    }
    return *one::MutJitEnabled();
  });
  m.def("set_jit_forward_args", [](const std::vector<std::shared_ptr<one::Tensor>>& tensors,
                                   const std::vector<std::shared_ptr<one::Tensor>>& parameters) {
    auto arg_tensors(tensors);
    for (const auto& p : parameters) { arg_tensors.push_back((p)); }
    SetJitForwardArgs(arg_tensors);
  });

  py::class_<jit::Module, std::shared_ptr<jit::Module>>(m, "JitModule")
      .def(py::init<py::object>())
      .def("__call__", &jit::Module::forward);
  ;
}

}  // namespace oneflow
#endif  // WITH_MLIR
