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
  CompTaskNode* GetRelatedFwTaskNode();

 protected:
  virtual void VirtualBuildExecGphAndBindOutDiffRegst() { UNEXPECTED_RUN(); }
  virtual void VirtualBuildActivationDiffRegst() { UNEXPECTED_RUN(); }
  virtual void VirtualBuildInDiffRegst() { UNEXPECTED_RUN(); }
  virtual void VirtualProduceInDiffAndBindEdge(TaskEdge* edge) {
    UNEXPECTED_RUN();
  };
  virtual void VirtualProduceRegstOnRecurrentEdge(TaskEdge* edge) {
    UNEXPECTED_RUN();
  }
  virtual void VirtualProduceActivationDiff() { UNEXPECTED_RUN(); }
  virtual void VirtualConsumeActivation(TaskEdge* edge) { UNEXPECTED_RUN(); }
  virtual void VirtualConsumeDiffRegst(TaskEdge* edge) { UNEXPECTED_RUN(); }
  virtual void VirtualConsumeInRegst() { UNEXPECTED_RUN(); };
  virtual void VirtualInferBlobDescInActivationDiff() { UNEXPECTED_RUN(); }
  virtual void VirtualInferBlobDescInHiddenDiff() { UNEXPECTED_RUN(); }

 private:
  void BindModelDiffRegst();
  void InferBlobDescsInProducedRegsts();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
