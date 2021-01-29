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
#include <pybind11/functional.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/instructions_builder.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  py::class_<InstructionsBuilder, std::shared_ptr<InstructionsBuilder>>(m, "InstructionsBuilder")
      .def(py::init([](const std::shared_ptr<vm::IdGenerator>& id_generator,
                       const std::shared_ptr<vm::cfg::InstructionListProto>& instruction_list,
                       const std::shared_ptr<eager::cfg::EagerSymbolList>& symbol_list,
                       const std::function<void(compatible_py::BlobObject*)>& release_object) {
        return std::make_shared<InstructionsBuilder>(id_generator, instruction_list, symbol_list,
                                                     release_object);
      }))
      .def("id_generator", &InstructionsBuilder::id_generator)
      .def("instruction_list", &InstructionsBuilder::instruction_list)
      .def("eager_symbol_list", &InstructionsBuilder::eager_symbol_list)
      .def("object_releaser", &InstructionsBuilder::object_releaser)
      .def("NewSymbolId",
           [](const std::shared_ptr<InstructionsBuilder>& x) {
             return x->NewSymbolId().GetOrThrow();
           })
      .def("GetSymbol4String",
           [](const std::shared_ptr<InstructionsBuilder>& x, std::string str) {
             return x->GetSymbol4String(str).GetPtrOrThrow();
           })
      .def("GetJobConfSymbol",
           [](const std::shared_ptr<InstructionsBuilder>& x,
              const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
             return x->GetJobConfSymbol(job_conf).GetPtrOrThrow();
           })
      .def("GetParallelDescSymbol",
           [](const std::shared_ptr<InstructionsBuilder>& x,
              const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
             return x->GetParallelDescSymbol(parallel_conf).GetPtrOrThrow();
           })
      .def("GetScopeSymbol", [](const std::shared_ptr<InstructionsBuilder>& x,
                                const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
        return x->GetScopeSymbol(scope_proto).GetPtrOrThrow();
      });

  // these API will be removed when InstructionsBuilder is refactor competely
  py::module_ vm_sub_module = m.def_submodule("vm");

  vm_sub_module.def("DelObjectOperand", &DelObjectOperand);
  vm_sub_module.def("MutOperand", &MutOperand);
  vm_sub_module.def("Int64Operand", &Int64Operand);

  vm_sub_module.def("InitSymbolOperand", &InitSymbolOperand);
  vm_sub_module.def("SymbolOperand", &SymbolOperand);
  vm_sub_module.def("ConstOperand", &ConstOperand);

  vm_sub_module.def("OperandSeparator", &OperandSeparator);
  vm_sub_module.def("Uint64Operand", &Uint64Operand);
  vm_sub_module.def("Mut2Operand", &Mut2Operand);
}

}  // namespace oneflow
