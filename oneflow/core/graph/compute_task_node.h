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
#ifndef ONEFLOW_CORE_GRAPH_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/graph/stream_index_getter_registry_manager.h"

namespace oneflow {

class LogicalNode;

class CompTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() = default;
  virtual ~CompTaskNode() = default;

  virtual CudaWorkType GetCudaWorkType() const {
#ifdef WITH_CUDA
    return CudaWorkType::kCompute;
#else
    UNIMPLEMENTED();
#endif
  }
  virtual void ToProto(TaskProto*) override;

  // parallel_ctx_
  int64_t parallel_id() const { return parallel_ctx_.parallel_id(); }
  const ParallelContext* parallel_ctx() const override { return &parallel_ctx_; }
  ParallelContext* mut_parallel_ctx() { return &parallel_ctx_; }

  // logical_node_
  const LogicalNode* logical_node() const { return logical_node_; }
  void set_logical_node(const LogicalNode* val) { logical_node_ = val; }
  std::string VisualStr() const override;

 protected:
  const LogicalNode* GetOneSuccLogicalNodeOnEdge(TaskEdge* edge);
  const LogicalNode* GetOnePredLogicalNodeOnEdge(TaskEdge* edge);
  std::vector<CompTaskNode*> GetSuccCompTaskNodesOnEdge(TaskEdge* edge) const;
  std::vector<CompTaskNode*> GetPredCompTaskNodesOnEdge(TaskEdge* edge) const;

  void InferProducedDataRegstTimeShape() override;

 private:
  ParallelContext parallel_ctx_;
  const LogicalNode* logical_node_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COMPUTE_TASK_NODE_H_
