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
#include "oneflow/core/job_rewriter/job_pass.h"

namespace oneflow {

namespace {

HashMap<std::string, const JobPass*>* PassName2JobPass() {
  static HashMap<std::string, const JobPass*> pass_name2job_pass;
  return &pass_name2job_pass;
}

}  // namespace

void RegisterJobPass(const std::string& pass_name, const JobPass* pass) {
  CHECK(PassName2JobPass()->emplace(pass_name, pass).second);
}

bool HasJobPass(const std::string& pass_name) {
  return PassName2JobPass()->find(pass_name) != PassName2JobPass()->end();
}

const JobPass& JobPass4Name(const std::string& pass_name) {
  const auto& iter = PassName2JobPass()->find(pass_name);
  CHECK(iter != PassName2JobPass()->end());
  return *iter->second;
}

}  // namespace oneflow
