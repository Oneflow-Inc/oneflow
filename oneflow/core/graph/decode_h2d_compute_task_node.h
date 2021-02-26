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
#ifndef ONEFLOW_CORE_GRAPH_DECODE_H2D_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_DECODE_H2D_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class DecodeH2DCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeH2DCompTaskNode);
  DecodeH2DCompTaskNode() = default;
  ~DecodeH2DCompTaskNode() override = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;

  TaskType GetTaskType() const override { return TaskType::kDecodeH2D; }
  CudaWorkType GetCudaWorkType() const override {
#ifdef WITH_CUDA
    return CudaWorkType::kDecodeH2D;
#else
    UNIMPLEMENTED();
#endif
  }

 private:
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_DECODE_H2D_COMPUTE_TASK_NODE_H_
