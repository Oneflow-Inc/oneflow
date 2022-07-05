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

#include <llvm/IR/IntrinsicsS390.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <tuple>
#include <vector>
#include "oneflow/core/common/singleton.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/py_ast.h"
#ifdef WITH_MLIR

#include "oneflow/ir/include/OneFlow/Extension.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowRoundTrip.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowAstJIT.h"
#include <glog/logging.h>
#include "oneflow/api/python/of_api_registry.h"
#include <functional>
#include <utility>

namespace oneflow {
ONEFLOW_API_PYBIND11_MODULE("ir", m) {
  m.def("load_jit_shared_lib",
        [](const std::string& lib_path) { MutSharedLibPaths()->insert(lib_path); });
  m.def("compile_and_register_lr_jit",
        [](const std::string& function_id
           //  ,const pyast::FunctionDef& func = pyast::FunctionDef::get_FunctionDef()) {
        ) {
          const pyast::FunctionDef& func = pyast::FunctionDef("test", {}, {});
          Singleton<LR_JIT>::New();
          Singleton<LR_JIT>::Get()->Register(function_id, func);
          auto engine = Singleton<LR_JIT>::Get()->LookUp(function_id);
          auto lr = Singleton<LR_JIT>::Get()->Invoke(engine, 1, 2);
          std::cout << lr << std::endl;
        });

  pybind11::class_<pyast::FunctionDef>(m, "FunctionDef");
  pybind11::class_<pyast::Return>(m, "Return");
  pybind11::class_<pyast::Assign>(m, "Assign");
  pybind11::class_<pyast::If>(m, "If");
  pybind11::class_<pyast::Raise>(m, "Raise");
  pybind11::class_<pyast::Assert>(m, "Assert");
  pybind11::class_<pyast::Expr>(m, "Expr");
  pybind11::class_<pyast::BoolOp>(m, "BoolOp");
  pybind11::class_<pyast::BinOp>(m, "BinOp");
  pybind11::class_<pyast::Lambda>(m, "Lambda");
  pybind11::class_<pyast::Compare>(m, "Compare");
  pybind11::class_<pyast::Call>(m, "Call");
  pybind11::class_<pyast::Num>(m, "Num");
  pybind11::class_<pyast::Constant>(m, "Constant");
  pybind11::class_<pyast::Attribute>(m, "Attribute");
  pybind11::class_<pyast::Name>(m, "Name");
  pybind11::class_<pyast::arguments>(m, "arguments");
  pybind11::class_<pyast::arg>(m, "arg");
}

}  // namespace oneflow

#endif  // WITH_MLIR
