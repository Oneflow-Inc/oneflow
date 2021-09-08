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
#ifndef ONEFLOW_CORE_STREAM_EXECUTION_CONTEXT_HOOK_H_
#define ONEFLOW_CORE_STREAM_EXECUTION_CONTEXT_HOOK_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {

class ExecutionContextHook {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecutionContextHook);
  ExecutionContextHook() = default;
  virtual ~ExecutionContextHook() = default;

  virtual Maybe<void> OnExecutionContextSetup() = 0;
  virtual Maybe<void> OnExecutionContextTeardown() = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_STREAM_EXECUTION_CONTEXT_HOOK_H_
