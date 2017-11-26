#ifndef ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class BackwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BackwardCompTaskNode);
  BackwardCompTaskNode() = default;
  ~BackwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

  TaskType GetTaskType() const override { return TaskType::kBackward; }

 private:
  using Lbn2NodeBnMap = HashMap<std::string, std::pair<ExecNode*, std::string>>;

  void BuildExecGphFromUserOps(Lbn2NodeBnMap* extern_in_lbn2consumer);
  void SetExecNodeFromOutdiffRegst(const Lbn2NodeBnMap& extern_in_lbn2consumer);
  void AddLbn2ActivationDiffRegst();
  void AddLbn2InDiffRegst();
  void AddLbn2ModelDiffRegst();
  void InferBlobDescsInProducedRegsts();
  std::shared_ptr<RegstDesc> GetRelatedInRegst();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BACKWARD_COMPUTE_TASK_NODE_H_
