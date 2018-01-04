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

 protected:
  virtual void BuildExecGphAndBindOutDiffRegst() { UNEXPECTED_RUN(); };
  virtual void BuildInDiffRegst() { UNEXPECTED_RUN(); };
  virtual void VirtualConsumeInRegst() { UNEXPECTED_RUN(); };
  TaskNode* GetRelatedFwTaskNode();

 private:
  void FixRegisterNumRange() override;
  void BuildActivationDiffRegst();
  void BuildModelDiffRegst();
  void InferBlobDescsInProducedRegsts();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
