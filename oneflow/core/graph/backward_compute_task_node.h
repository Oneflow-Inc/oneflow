#ifndef ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class BackwardCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BackwardCompTaskNode);
  BackwardCompTaskNode() = default;
  virtual ~BackwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

 private:
  void BuildExecGphAndBindOutDiffRegst();
  void BuildActivationDiffRegst();
  void BuildInDiffRegst();
  void BuildModelDiffRegst();
  void InferBlobDescsInProducedRegsts();
  std::shared_ptr<RegstDesc> GetRelatedInRegst();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
