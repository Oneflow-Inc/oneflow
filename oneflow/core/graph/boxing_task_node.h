#ifndef ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class BoxingTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  ~BoxingTaskNode() = default;

  void Init(int64_t machine_id,
            std::function<void(BoxingOpConf*)> BoxingOpConfSetter);
  void NewAllProducedRegst() override { TODO(); }
  TodoTaskType GetTaskType() const override { return TodoTaskType::kBoxing; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_
