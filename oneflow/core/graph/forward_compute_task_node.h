#ifndef ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ForwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForwardCompTaskNode);
  ForwardCompTaskNode() = default;
  ~ForwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;
  void LockRegsts() override;
  bool IsReadyForBuild() override;

  TaskType GetTaskType() const override { return TaskType::kForward; }

 private:
  using Lbn2NodeBnMap = HashMap<std::string, std::pair<ExecNode*, std::string>>;

  void BuildFromUserOps(Lbn2NodeBnMap* lbn2producer,
                        Lbn2NodeBnMap* lbn2consumer,
                        Lbn2NodeBnMap* extern_in_lbn2consumer);
  void SetExecNodeFromInRegst(const Lbn2NodeBnMap& extern_in_lbn2consumer);
  void AddLbn2OutRegst(const Lbn2NodeBnMap& lbn2consumer);
  void AddLbn2ActivationRegst();
  void AddLbn2ModelAndTmpRegsts();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_FORWARD_COMPUTE_TASK_NODE_H_
