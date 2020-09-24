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
#include "oneflow/core/job_rewriter/op_graph_pass.h"

namespace oneflow {

namespace {

HashMap<std::string, const OpGraphPass*>* PassName2FunctionPass() {
  static HashMap<std::string, const OpGraphPass*> pass_name2job_pass;
  return &pass_name2job_pass;
}

}  // namespace

void RegisterFunctionPass(const std::string& pass_name, const OpGraphPass* pass) {
  CHECK(PassName2FunctionPass()->emplace(pass_name, pass).second);
}

bool HasFunctionPass(const std::string& pass_name) {
  return PassName2FunctionPass()->find(pass_name) != PassName2FunctionPass()->end();
}

const OpGraphPass& FunctionPass(const std::string& pass_name) {
  const auto& iter = PassName2FunctionPass()->find(pass_name);
  CHECK(iter != PassName2FunctionPass()->end());
  return *iter->second;
}

}  // namespace oneflow
