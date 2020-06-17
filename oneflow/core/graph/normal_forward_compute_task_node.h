#ifndef ONEFLOW_CORE_GRAPH_NORMAL_FORWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NORMAL_FORWARD_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class NormalForwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalForwardCompTaskNode);
  NormalForwardCompTaskNode() = default;
  ~NormalForwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  bool IsReadyForBuild() override;

  TaskType GetTaskType() const override { return TaskType::kNormalForward; }
  bool HasBackwardCompTaskNode();

 private:
  bool IsAllOutNodeNormalForward() const;
  bool CanProduceSeperatedRegstsForEachOutBlob() const;
  void ProduceOutRegstByNameAndBlockNum(const std::string& name, size_t mem_block_num);
  void BuildExecGphAndRegst() override;
  void BuildExecGphStructAndBindInRegst();
  void BuildOutRegst();
  void BuildTmp7BufRegsts();
  void InferProducedDataRegstTimeShape() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NORMAL_FORWARD_COMPUTE_TASK_NODE_H_
