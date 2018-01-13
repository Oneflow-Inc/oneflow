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
  TaskNode* GetRelatedFwTaskNode();

 protected:
  virtual void VirtualBuildExecGphAndBindOutDiffRegst() { UNEXPECTED_RUN(); }
  virtual void VirtualBindActivationDiffRegst() {}
  virtual void VirtualBindInDiffRegst() { UNEXPECTED_RUN(); }
  virtual void VirtualProduceInDiffAndBindEdge(TaskEdge* edge) {
    UNEXPECTED_RUN();
  };
  virtual void VirtualProduceRegstOnSelfEdge(TaskEdge* edge) {
    UNEXPECTED_RUN();
  }
  virtual void VirtualProduceActivationDiff() {}
  virtual void VirtualConsumeActivation(TaskEdge* edge) {}
  virtual void VirtualConsumeDiffRegst(TaskEdge* edge) { UNEXPECTED_RUN(); }
  virtual void VirtualConsumeInRegst() { UNEXPECTED_RUN(); };
  virtual void VirtualInferBlobDescInActivationDiff() {}
  virtual void VirtualInferBlobDescInHiddenDiff() {}

 private:
  void BindModelDiffRegst();
  void InferBlobDescsInProducedRegsts();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
