#ifndef ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

class MdSaveCompTaskNode final : public SinkCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompTaskNode);
  MdSaveCompTaskNode() = default;
  ~MdSaveCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kMdSave; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMPUTE_TASK_NODE_H_
