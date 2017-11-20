#ifndef ONEFLOW_CORE_GRAPH_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class ChainNode;

class CompTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() = default;
  virtual ~CompTaskNode() = default;

  virtual void FixThrdLocId() {}
  virtual void ToProto(TaskProto*) override;

  // parallel_ctx_
  int64_t parallel_id() const { return parallel_ctx_.parallel_id(); }
  const ParallelContext* parallel_ctx() const override {
    return &parallel_ctx_;
  }
  ParallelContext* mut_parallel_ctx() { return &parallel_ctx_; }

  // chain_node_
  const ChainNode* chain_node() const { return chain_node_; }
  void set_chain_node(const ChainNode* val) { chain_node_ = val; }

 protected:
 private:
  ParallelContext parallel_ctx_;
  const ChainNode* chain_node_;
};

void SortByParallelId(std::vector<CompTaskNode*>* node_vec);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COMPUTE_TASK_NODE_H_
