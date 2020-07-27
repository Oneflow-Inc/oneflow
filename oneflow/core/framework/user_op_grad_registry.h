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
#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_GRAD_REGISTRY_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_GRAD_REGISTRY_H_

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

using AddOpFn = std::function<void(const UserOpConfWrapper&)>;
using GenBackwardOpConfFn = std::function<void(const UserOpWrapper&, AddOpFn)>;
using BackwardOpConfGenFn = std::function<void(BackwardOpConfContext*)>;

struct OpGradRegistryResult {
  std::string op_type_name;
  GenBackwardOpConfFn gen_bw_fn;
  BackwardOpConfGenFn bw_gen_fn;
};

class OpGradRegistry final {
 public:
  OpGradRegistry& Name(const std::string& op_type_name);
  // old
  OpGradRegistry& SetGenBackwardOpConfFn(GenBackwardOpConfFn fn);
  // new
  OpGradRegistry& SetBackwardOpConfGenFn(BackwardOpConfGenFn fn);

  OpGradRegistry& Finish();
  OpGradRegistryResult GetResult() { return result_; }

 private:
  OpGradRegistryResult result_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_GRAD_REGISTRY_H_
