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
#ifndef ONEFLOW_CORE_GRAPH_DISTRIBUTE_SPLIT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_DISTRIBUTE_SPLIT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class DistributeSplitCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeSplitCompTaskNode);
  DistributeSplitCompTaskNode() = default;
  ~DistributeSplitCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  bool IsReadyForBuild() override;

  TaskType GetTaskType() const override { return TaskType::kDistributeSplit; }
  bool HasBackwardCompTaskNode();

 private:
  void BuildExecGphAndRegst() override;
  void BuildExecGphStructAndBindInRegst();
  void BuildOutRegst();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_DISTRIBUTE_SPLIT_COMPUTE_TASK_NODE_H_
