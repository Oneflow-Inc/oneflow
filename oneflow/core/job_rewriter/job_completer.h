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
#ifndef ONEFLOW_CORE_JOB_REWRITER_JOB_COMPLETER_H_
#define ONEFLOW_CORE_JOB_REWRITER_JOB_COMPLETER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/framework/tensor.h"

namespace oneflow {

class JobCompleter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobCompleter);
  JobCompleter() = default;
  ~JobCompleter() = default;

  static Maybe<void> Complete(Job* job);
  // The job is copied from a shared graph, it needs to be modified
  // for a new graph with different input.
  static Maybe<void> UpdateSharedGraphForNewInput(
      Job* job,
      const std::function<Maybe<std::shared_ptr<one::Tensor>>(const std::string&)>&
          InputTensor4Name,
      const std::function<Maybe<const OperatorConf*>(const std::string& shared_op_name)>&
          NewOp4SharedOpName);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_JOB_COMPLETER_H_
