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
  virtual void VirtualBuildExecGphAndBindOutDiffRegst() { UNIMPLEMENTED(); }
  virtual void VirtualBuildActivationDiffRegst() {}
  virtual void VirtualBuildInDiffRegst() { UNIMPLEMENTED(); }
  virtual void VirtualBuildExtraRegsts() {}
  virtual void VirtualProduceInDiffAndBindEdge(TaskEdge* edge) {
    UNIMPLEMENTED();
  };
  virtual void VirtualProduceRegstOnRecurrentEdge(TaskEdge* edge) {
    UNIMPLEMENTED();
  }
  virtual void VirtualProduceActivationDiff() {}
  virtual void VirtualConsumeActivation(TaskEdge* edge) {}
  virtual void VirtualConsumeRegstOnInEdge(TaskEdge* edge) { UNIMPLEMENTED(); }
  virtual void VirtualConsumeInRegst() { UNIMPLEMENTED(); }
  virtual void VirtualInferBlobDescInActivationDiff() {}
  virtual void VirtualInferBlobDescInHiddenDiff() {}

 private:
  void LinkFwExecNode();
  void BindModelDiffRegst();
  void InferBlobDescsInProducedRegsts();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
