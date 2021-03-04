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
#ifndef ONEFLOW_CORE_GRAPH_COLLECTIVE_BOXING_UNPACK_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COLLECTIVE_BOXING_UNPACK_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CollectiveBoxingUnpackTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingUnpackTaskNode);
  CollectiveBoxingUnpackTaskNode() = default;
  ~CollectiveBoxingUnpackTaskNode() override = default;

  void Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
            const Shape& logical_shape, const SbpParallel& src_sbp_parallel,
            const SbpParallel& dst_sbp_parallel, const int64_t parallel_num);

  TaskType GetTaskType() const override { return TaskType::kCollectiveBoxingUnpack; }

 private:
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() final;
  void InferProducedDataRegstTimeShape() final;

  LogicalBlobId lbi_;
  Shape logical_shape_;
  SbpParallel src_sbp_parallel_;
  SbpParallel dst_sbp_parallel_;
  int64_t parallel_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COLLECTIVE_BOXING_UNPACK_TASK_NODE_H_
