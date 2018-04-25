#ifndef ONEFLOW_CORE_GRAPH_NORMAL_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMAL_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class NormalBackwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalBackwardCompTaskNode);
  NormalBackwardCompTaskNode() = default;
  ~NormalBackwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  CompTaskNode* GetRelatedFwTaskNode();

 protected:
  void BuildExecGphAndBindOutDiffRegst();
  void BuildActivationDiffRegst();
  void BuildInDiffRegst();

 private:
  void LinkFwExecNode();
  void BindModelDiffRegst();
  void InferBlobDescsInProducedRegsts();

  std::list<std::weak_ptr<RegstDesc>> in_regst_boxing_;
  std::list<std::weak_ptr<RegstDesc>> in_regst_121_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMAL_BACKWARD_COMPUTE_TASK_NODE_H_
