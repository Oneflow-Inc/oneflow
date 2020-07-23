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
#ifndef ONEFLOW_CORE_GRAPH_SLICE_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_SLICE_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

enum SliceBoxingTaskMode {
  kSliceBoxingTaskModeInvalid,
  kSliceBoxingTaskModeCopy,
  kSliceBoxingTaskModeAdd,
};

class SliceBoxingTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingTaskNode);
  SliceBoxingTaskNode() = default;
  ~SliceBoxingTaskNode() override = default;

  void Init(const LogicalBlobId& lbi, const TensorSliceView& out_slice, SliceBoxingTaskMode mode,
            int64_t machine_id, int64_t thrd_id, int64_t mem_zone_id);
  void Init(const LogicalBlobId& lbi, const TensorSliceView& out_slice, SliceBoxingTaskMode mode,
            int64_t machine_id, int64_t thrd_id);
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  TaskType GetTaskType() const override { return TaskType::kSliceBoxing; }
  void SetInDataEdgeSlice(const TaskEdge* edge, const TensorSliceView& slice);
  void ConnectToSrcNodeWithSlice(TaskNode* src, TaskEdge* edge, const TensorSliceView& slice);
  void SetOutShape(const Shape& shape);

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
  OperatorConf GetBoxingOpConf();
  void InitProducedRegstMemCase(MemoryCase*) override;

  HashMap<const TaskEdge*, TensorSliceView> in_data_edge2slice_;
  std::vector<const TaskEdge*> ordered_in_data_edges_;
  LogicalBlobId lbi_;
  TensorSliceView out_slice_;
  Shape out_shape_;
  SliceBoxingTaskMode mode_ = kSliceBoxingTaskModeInvalid;
  int64_t mem_zone_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SLICE_BOXING_TASK_NODE_H_
