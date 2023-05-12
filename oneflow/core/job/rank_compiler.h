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
#ifndef ONEFLOW_CORE_JOB_RANK_COMPILER_H_
#define ONEFLOW_CORE_JOB_RANK_COMPILER_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/boxing_task_graph.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RankCompiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RankCompiler);
  RankCompiler(const std::shared_ptr<BoxingTaskGraphProto>& boxing_task_graph_proto, int64_t rank)
      : boxing_task_graph_proto_(boxing_task_graph_proto), rank_(rank) {}
  ~RankCompiler() = default;

  Maybe<void> Compile(const HashSet<std::string>& var_op_names, Job* job, Plan* plan) const;

 private:
  std::shared_ptr<BoxingTaskGraphProto> boxing_task_graph_proto_;
  int64_t rank_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RANK_COMPILER_H_
