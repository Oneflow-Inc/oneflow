#ifndef ONEFLOW_CORE_GRAPH_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COMP_TASK_NODE_H_

#include <algorithm>
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CompTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CompTaskNode);
  CompTaskNode() = default;
  virtual ~CompTaskNode() = default;

  // Getters and Setters
  uint64_t parallel_id() const { return parallel_id_; }
  void set_parallel_id(uint64_t parallel_id) { parallel_id_ = parallel_id; }
  bool IsLossNode() const { return chain_node()->IsLossNode(); }
  std::string VisualStr() const override;
  virtual void ToProto(TaskProto* ret) const override {
    TaskNode::ToProto(ret);
    ret->set_parallel_id(parallel_id_);
  }
  std::string device_name() const;

 protected:
  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
    auto fw_comp_code = of_dynamic_cast<CompTaskNode*> (fw_node);
    parallel_id_ = fw_comp_code->parallel_id_;
  }

 private:
  uint64_t parallel_id_;

};

void SortByParallelId(std::vector<CompTaskNode*>* comp_node_vec);

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_COMP_TASK_NODE_H_
