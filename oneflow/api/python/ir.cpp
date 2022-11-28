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
#include "oneflow/core/common/singleton.h"
#include "oneflow/ir/oneflow-extension/include/PyAst/Ast.h"

#include <llvm/IR/IntrinsicsS390.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <tuple>
#include <vector>

#ifdef WITH_MLIR

#include "oneflow/ir/include/OneFlow/Extension.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowRoundTrip.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowLRJITRegistry.h"
#include "oneflow/api/python/of_api_registry.h"
#include <glog/logging.h>
#include <functional>
#include <utility>

namespace oneflow {
ONEFLOW_API_PYBIND11_MODULE("ir", m) {
  m.def("load_jit_shared_lib",
        [](const std::string& lib_path) { MutSharedLibPaths()->insert(lib_path); });

  // TODO: this may be move to a common place for create global singleton.
  m.def("create_global_lr_jit", []() { Singleton<LRJITRegistry>::New(); });

  m.def("compile_and_register_lr_jit", [](const std::string& function_id,
                                          std::shared_ptr<pyast::FunctionDef>& func, bool is_dump) {
    Singleton<LRJITRegistry>::Get()->Register(function_id, *func.get(), is_dump);
  });

  // look up and execute the registered function for python api
  m.def("get_lr", [](const std::string& function_id, float base_lr, float step) {
    auto engine = Singleton<LRJITRegistry>::Get()->LookUp(function_id);
    return engine(base_lr, step);
  });

  pybind11::class_<pyast::stmt, std::shared_ptr<pyast::stmt>>(m, "smt");

  pybind11::class_<pyast::expr, std::shared_ptr<pyast::expr>>(m, "expr");

  pybind11::class_<pyast::FunctionDef, pyast::stmt, std::shared_ptr<pyast::FunctionDef>>(
      m, "FunctionDef");
  m.def("FunctionDef_", &pyast::FunctionDef::FunctionDef_);

  pybind11::class_<pyast::Return, pyast::stmt, std::shared_ptr<pyast::Return>>(m, "Return");
  m.def("Return_", &pyast::Return::Return_);

  pybind11::class_<pyast::Assign, pyast::stmt, std::shared_ptr<pyast::Assign>>(m, "Assign");
  m.def("Assign_", &pyast::Assign::Assign_);

  pybind11::class_<pyast::If, pyast::stmt, std::shared_ptr<pyast::If>>(m, "If");
  m.def("If_", &pyast::If::If_);

  pybind11::class_<pyast::Raise, pyast::stmt, std::shared_ptr<pyast::Raise>>(m, "Raise");
  m.def("Raise_", &pyast::Raise::Raise_);

  pybind11::class_<pyast::Assert, pyast::stmt, std::shared_ptr<pyast::Assert>>(m, "Assert");
  m.def("Assert_", &pyast::Assert::Assert_);

  pybind11::class_<pyast::Expr, pyast::stmt, std::shared_ptr<pyast::Expr>>(m, "Expr");
  m.def("Expr_", &pyast::Expr::Expr_);

  pybind11::class_<pyast::BoolOp, pyast::expr, std::shared_ptr<pyast::BoolOp>>(m, "BoolOp");
  m.def("BoolOp_", &pyast::BoolOp::BoolOp_);

  pybind11::class_<pyast::BinOp, pyast::expr, std::shared_ptr<pyast::BinOp>>(m, "BinOp");
  m.def("BinOp_", &pyast::BinOp::BinOp_);

  pybind11::class_<pyast::Lambda, pyast::expr, std::shared_ptr<pyast::Lambda>>(m, "Lambda");
  m.def("Lambda_", &pyast::Lambda::Lambda_);

  pybind11::class_<pyast::Compare, pyast::expr, std::shared_ptr<pyast::Compare>>(m, "Compare");
  m.def("Compare_", &pyast::Compare::Compare_);

  pybind11::class_<pyast::Call, pyast::expr, std::shared_ptr<pyast::Call>>(m, "Call");
  m.def("Call_", &pyast::Call::Call_);

  pybind11::class_<pyast::Num, pyast::expr, std::shared_ptr<pyast::Num>>(m, "Num");
  m.def("Num_", &pyast::Num::Num_);

  pybind11::class_<pyast::Constant, pyast::expr, std::shared_ptr<pyast::Constant>>(m, "Constant");
  m.def("Constant_", &pyast::Constant::Constant_);

  pybind11::class_<pyast::Attribute, pyast::expr, std::shared_ptr<pyast::Attribute>>(m,
                                                                                     "Attribute");
  m.def("Attribute_", &pyast::Attribute::Attribute_);

  pybind11::class_<pyast::Name, pyast::expr, std::shared_ptr<pyast::Name>>(m, "Name");
  m.def("Name_", &pyast::Name::Name_);

  pybind11::class_<pyast::arguments, std::shared_ptr<pyast::arguments>>(m, "arguments");
  m.def("arguments_", &pyast::arguments::arguments_);

  pybind11::class_<pyast::arg, std::shared_ptr<pyast::arg>>(m, "arg");
  m.def("arg_", &pyast::arg::arg_);
}

}  // namespace oneflow

#endif  // WITH_MLIR
