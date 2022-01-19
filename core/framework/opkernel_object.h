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
#ifndef ONEFLOW_CORE_FRAMEWORK_OPKERNEL_OBJECT_H_
#define ONEFLOW_CORE_FRAMEWORK_OPKERNEL_OBJECT_H_

#include <functional>
#include "oneflow/core/framework/object.h"
#include "oneflow/core/operator/op_conf.cfg.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {

namespace compatible_py {

class OpKernelObject : public Object {
 public:
  OpKernelObject(int64_t object_id, const std::shared_ptr<cfg::OperatorConf>& op_conf,
                 const std::function<void(Object*)>& release);
  ~OpKernelObject() override { ForceReleaseAll(); }

  std::shared_ptr<cfg::OperatorConf> op_conf() const { return op_conf_; }
  std::shared_ptr<Scope> scope_symbol() const { return scope_symbol_; }

 private:
  void ForceReleaseAll();

  std::shared_ptr<cfg::OperatorConf> op_conf_;
  std::shared_ptr<Scope> scope_symbol_;
  std::vector<std::function<void(Object*)>> release_;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OBJECT_H_
