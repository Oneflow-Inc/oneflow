#ifndef ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class MdSaveCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompTaskNode);
  MdSaveCompTaskNode() = default;
  ~MdSaveCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  TodoTaskType GetTaskType() const override { return TodoTaskType::kMdSave; }
  void FixThrdLocId() override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMPUTE_TASK_NODE_H_
