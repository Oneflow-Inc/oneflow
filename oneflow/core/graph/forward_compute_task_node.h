#ifndef ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ForwardCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardCompTaskNode);
  ForwardCompTaskNode() : random_seed_(-1) {}
  virtual ~ForwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  void LockRegsts() override;

  void set_random_seed(int64_t random_seed) { random_seed_ = random_seed; }
  virtual void ToProto(TaskProto*) override;

 protected:
  virtual void VirtualAddRegstOnRecurrentOutEdge(TaskEdge* edge);
  virtual void VirtualConsumeRegstOnInEdge(TaskEdge* edge) { UNIMPLEMENTED(); }
  virtual void VirtualProduceRegstOnOutEdge(TaskEdge* edge);
  virtual void VirtualLockExtraRegsts() {}
  virtual void VirtualBuildExtraRegsts() {}
  virtual void VirtualBuildExecGphStructAndBindInRegst() { UNIMPLEMENTED(); }
  virtual void VirtualBuildOutRegst() { UNIMPLEMENTED(); }

 private:
  void BuildActivationRegst();
  void BuildModelAndTmpRegsts();

  int64_t random_seed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
