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
#ifdef WITH_MLIR

#include "oneflow/ir/include/OneFlow/Extension.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowRoundTrip.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowAstJIT.h"
#include <glog/logging.h>
#include "oneflow/api/python/of_api_registry.h"
#include <functional>
#include <utility>

class PyASTNode {
  pybind11::object _node;

 public:
  explicit PyASTNode(pybind11::object node) : _node(std::move(node)){};
  std::string GetName() {
    auto cls = _node.attr("__class__");
    auto name = cls.attr("__name__");
    return name.cast<pybind11::str>();
  }

  std::vector<std::string> GetFields() {
    auto fields = _node.attr("_fields");
    std::vector<std::string> res;
    std::for_each(fields.begin(), fields.end(),
                  [&res](pybind11::handle field) { res.push_back(field.cast<pybind11::str>()); });
    return res;
  }

  PyASTNode Visit(const std::string& name) { return PyASTNode(_node.attr(name.c_str())); }

  std::vector<PyASTNode> AsList() {
    std::vector<PyASTNode> res;
    for (auto item : _node) {
      auto node = PyASTNode(item.cast<pybind11::object>());
      res.push_back(node);
    }
    return res;
  }

  int AsInt() { return _node.cast<pybind11::int_>(); }

  std::string AsStr() { return _node.cast<pybind11::str>(); }

  float AsFloat() { return _node.cast<pybind11::float_>(); }
};

std::string PyASTNodeWrapper::GetName() { return _node->GetName(); }

std::vector<std::string> PyASTNodeWrapper::GetFields() { return _node->GetFields(); }

PyASTNodeWrapper PyASTNodeWrapper::Visit(const std::string& name) {
  auto _py_ast_node = std::make_shared<PyASTNode>(_node->Visit(name));
  return PyASTNodeWrapper(_py_ast_node);
}

int PyASTNodeWrapper::AsInt() { return _node->AsInt(); }

std::string PyASTNodeWrapper::AsStr() { return _node->AsStr(); }

float PyASTNodeWrapper::AsFloat() { return _node->AsFloat(); }

std::vector<PyASTNodeWrapper> PyASTNodeWrapper::AsList() {
  std::vector<PyASTNodeWrapper> res;
  auto list = _node->AsList();
  for (auto item : list) {
    auto item_wrapper = PyASTNodeWrapper(std::make_shared<PyASTNode>(item));
    res.push_back(item_wrapper);
  }
  return res;
}



namespace oneflow {
ONEFLOW_API_PYBIND11_MODULE("ir", m) {
  m.def("load_jit_shared_lib",
        [](const std::string& lib_path) { MutSharedLibPaths()->insert(lib_path); });
  m.def("compile_and_register_lr_jit",
        [](const pybind11::object& ast, const std::string& function_id) {
          Singleton<LR_JIT>::New();
          auto _py_ast_node = std::make_shared<PyASTNode>(ast);
          auto _py_ast_node_wrapper = PyASTNodeWrapper(_py_ast_node);
          Singleton<LR_JIT>::Get()->Register(function_id, _py_ast_node_wrapper);
          auto engine = Singleton<LR_JIT>::Get()->LookUp(function_id);
          auto lr = Singleton<LR_JIT>::Get()->Invoke(engine, 1, 2);
          std::cout << lr << std::endl;
        });
}

}  // namespace oneflow

#endif  // WITH_MLIR
