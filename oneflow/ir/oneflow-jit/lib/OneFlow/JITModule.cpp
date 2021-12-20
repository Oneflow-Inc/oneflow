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
#include "oneflow/ir/include/OneFlow/Extension.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowRoundTrip.h"
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
  std::shared_ptr<one::Tensor> forward(const py::args& args, const py::kwargs& kwargs) {
    auto jit_interpreter =
        std::dynamic_pointer_cast<one::JitInterpreter>(one::JitInterpreter::Get());
    CHECK(jit_interpreter != nullptr) << "JIT interpreter is not initialized";
    auto i_this = reinterpret_cast<std::uintptr_t>(this);
    std::string func_name = "jitModule" + std::to_string(i_this);
    auto inputs = args.cast<std::vector<std::shared_ptr<one::Tensor>>>();
    auto parameters_generator = py_module_.attr("parameters")();
    std::vector<std::shared_ptr<one::Tensor>> parameters{};
    for (auto p : parameters_generator) {
      parameters.push_back(p.cast<std::shared_ptr<one::Tensor>>());
    }
    std::vector<std::shared_ptr<one::Tensor>> arg_tensors{};
    arg_tensors.insert(arg_tensors.end(), inputs.begin(), inputs.end());
    arg_tensors.insert(arg_tensors.end(), parameters.begin(), parameters.end());
    std::vector<std::shared_ptr<one::Tensor>> tensors_to_materialize{};
    jit_interpreter->MarkMlirTraceStart();
    if (!func_op_) {
      func_op_ = jit_interpreter->Trace(importer_, func_name, arg_tensors, [&]() {
        auto returned_obj = py_module_.attr("forward")(*args, **kwargs);
        if (auto tensor = returned_obj.cast<std::shared_ptr<one::Tensor>>()) {
          importer_.FinalizeProcessFunction(tensor);
        } else {
          LOG(FATAL) << "return type not supported: " << returned_obj.get_type();
        }
      });
    }
    jit_interpreter->MarkMlirTraceEnd();
    auto ret = jit_interpreter->DispatchFunc(func_op_.getValue(), arg_tensors);
    jit_interpreter->MarkMlirDispatchEnd();
    LOG(ERROR) << "JIT trace overhead: " << jit_interpreter->TraceOverhead();
    return ret;
  }
  void dump_ir() { module_->dump(); }

 private:
  py::object py_module_;
  MLIRContext* context_;
  OwningOpRef<ModuleOp> module_;
  mlir::oneflow::JitImporter importer_;
  llvm::Optional<FuncOp> func_op_;
};
}  // namespace jit

ONEFLOW_API_PYBIND11_MODULE("jit", m) {
  m.def("toggle_jit", [](const std::string& func_name) {
    *one::MutJitEnabled() = !*one::MutJitEnabled();
    // when false => true, start jit
    auto jit_interpreter =
        std::dynamic_pointer_cast<one::JitInterpreter>(one::JitInterpreter::Get());
    CHECK(jit_interpreter != nullptr) << "JIT interpreter is not initialized";
    if (one::IsJitEnabled() == true) { jit_interpreter->MarkMlirTraceStart(); }
    // when true => false, start exec
    if (one::IsJitEnabled() == false) {
      jit_interpreter->Interrupt();
      jit_interpreter->MarkMlirTraceEnd();
      if (ParseBooleanFromEnv("ONEFLOW_MLIR_ENABLE_TRACE_PROFILING", false)) {
        LOG(ERROR) << "JIT trace overhead: " << jit_interpreter->TraceOverhead();
      }
    }
    return *one::MutJitEnabled();
  });
  m.def("set_jit_forward_args", [](const std::vector<std::shared_ptr<one::Tensor>>& tensors,
                                   const std::vector<std::shared_ptr<one::Tensor>>& parameters) {
    auto arg_tensors(tensors);
    for (const auto& p : parameters) { arg_tensors.push_back((p)); }
    TODO() << "set global fw args";
  });

  py::class_<jit::Module, std::shared_ptr<jit::Module>>(m, "JitModule")
      .def(py::init<py::object>())
      .def("__call__", &jit::Module::forward)
      .def("dump_ir", &jit::Module::dump_ir);
}

}  // namespace oneflow

COMMAND(SetJitInterpreter(::oneflow::one::JitInterpreter::Get()));

#endif  // WITH_MLIR
