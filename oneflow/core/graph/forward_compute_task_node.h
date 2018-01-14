#ifndef ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ForwardCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardCompTaskNode);
  ForwardCompTaskNode() = default;
  virtual ~ForwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  void LockRegsts() override;

 protected:
  virtual void VirtualAddRegstForRecurrentOutEdge(TaskEdge* edge) {}
  virtual void VirtualConsumeInRegst(TaskEdge* edge) { UNEXPECTED_RUN(); }
  virtual void BuildExecGphStructAndBindInRegst() { UNEXPECTED_RUN(); }
  virtual void VirtualBindOutRegst() {}
  virtual void VirtualFixRegisterNumRange() {}

 private:
  void BindOutRegst();
  void BindActivationRegst();
  void BindModelAndTmpRegsts();
  void FixRegisterNumRange() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
