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
#ifndef ONEFLOW_CORE_JOB_COMPILER_H_
#define ONEFLOW_CORE_JOB_COMPILER_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class Compiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Compiler);
  Compiler() = default;
  ~Compiler() = default;

  void Compile(Job*, Plan*) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPILER_H_
