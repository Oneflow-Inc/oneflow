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
#include "oneflow/core/framework/user_op_grad_registry.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

OpGradRegistry& OpGradRegistry::Name(const std::string& op_type_name) {
  CHECK(!op_type_name.empty());
  result_.op_type_name = op_type_name;
  return *this;
}

OpGradRegistry& OpGradRegistry::SetGenBackwardOpConfFn(GenBackwardOpConfFn fn) {
  result_.gen_bw_fn = std::move(fn);
  return *this;
}

OpGradRegistry& OpGradRegistry::SetBackwardOpConfGenFn(BackwardOpConfGenFn fn) {
  result_.bw_gen_fn = std::move(fn);
  return *this;
}

OpGradRegistry& OpGradRegistry::Finish() {
  CHECK((result_.gen_bw_fn != nullptr) || (result_.bw_gen_fn != nullptr))
      << "No BackwardOpConf generate function for " << result_.op_type_name;
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
