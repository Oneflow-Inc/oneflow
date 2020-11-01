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

using PassName2Creator = HashMap<std::string, std::function<OpGraphPass*(const JobDesc&)>>;

PassName2Creator* GetPassName2Creator() {
  static PassName2Creator pass_name2creator;
  return &pass_name2creator;
}

}  // namespace

void RegisterFunctionPass(const std::string& pass_name,
                          const std::function<OpGraphPass*(const JobDesc&)>& pass_creator) {
  CHECK(GetPassName2Creator()->emplace(pass_name, pass_creator).second);
}

bool HasFunctionPass(const std::string& pass_name) {
  return GetPassName2Creator()->find(pass_name) != GetPassName2Creator()->end();
}

std::unique_ptr<const OpGraphPass> FunctionPass(const std::string& pass_name,
                                                const JobDesc& job_desc) {
  const auto& iter = GetPassName2Creator()->find(pass_name);
  CHECK(iter != GetPassName2Creator()->end());
  const auto& Creator = iter->second;
  return std::unique_ptr<const OpGraphPass>(Creator(job_desc));
}

}  // namespace oneflow
