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
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/graph/stream_index_getter_registry_manager.h"

namespace oneflow {

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

  // op_node_
  const OpNode* op_node() const { return op_node_; }
  void set_op_node(const OpNode* val) { op_node_ = val; }
  std::string VisualStr() const override;

  // op
  std::shared_ptr<const Operator> op() const { return op_node_->shared_op(); }

 protected:
  const OpNode* GetOneSuccOpNodeOnEdge(TaskEdge* edge);
  const OpNode* GetOnePredOpNodeOnEdge(TaskEdge* edge);
  std::vector<CompTaskNode*> GetSuccCompTaskNodesOnEdge(TaskEdge* edge) const;
  std::vector<CompTaskNode*> GetPredCompTaskNodesOnEdge(TaskEdge* edge) const;

  void InferProducedDataRegstTimeShape() override;

 private:
  ParallelContext parallel_ctx_;
  const OpNode* op_node_;
};

class OpCompTaskNodeCreator {
 public:
  virtual ~OpCompTaskNodeCreator() = default;
  virtual CompTaskNode* NewCompTaskNode(const OperatorConf& op_conf) = 0;
};

template<typename CompTaskNodeType>
class StaticOpCompTaskNodeCreator : public OpCompTaskNodeCreator {
 public:
  StaticOpCompTaskNodeCreator() = default;
  ~StaticOpCompTaskNodeCreator() override = default;

 private:
  CompTaskNode* NewCompTaskNode(const OperatorConf& op_conf) override {
    return new CompTaskNodeType();
  }
};

class FnOpCompTaskNodeCreator : public OpCompTaskNodeCreator {
 public:
  using CreateFn = std::function<CompTaskNode*(const OperatorConf& op_conf)>;
  explicit FnOpCompTaskNodeCreator(CreateFn fn) : fn_(std::move(fn)) {}
  ~FnOpCompTaskNodeCreator() override = default;

 private:
  CompTaskNode* NewCompTaskNode(const OperatorConf& op_conf) override { return fn_(op_conf); }
  CreateFn fn_;
};

#define REGISTER_USER_OP_COMP_TASK_NODE_TYPE(op_type_name, comp_task_node_type) \
  REGISTER_CLASS_CREATOR(std::string, op_type_name, OpCompTaskNodeCreator,      \
                         ([] { return new StaticOpCompTaskNodeCreator<comp_task_node_type>(); }));

#define REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE(op_type_case, comp_task_node_type) \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, OpCompTaskNodeCreator,            \
                         ([] { return new StaticOpCompTaskNodeCreator<comp_task_node_type>(); }));

#define REGISTER_SYSTEM_OP_COMP_TASK_NODE_TYPE_WITH_FUNC(op_type_case, func) \
  REGISTER_CLASS_CREATOR(int32_t, op_type_case, OpCompTaskNodeCreator,       \
                         ([] { return new FnOpCompTaskNodeCreator(func); }));

CompTaskNode* NewCompTaskNode4OpNode(const OpNode* op_node);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COMPUTE_TASK_NODE_H_
