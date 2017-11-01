#ifndef ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class BoxingTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  virtual ~BoxingTaskNode() = default;

  void Init(int64_t machine_id);
  void ProduceAllRegstsAndBindEdges() override { TODO(); }
  TodoTaskType GetTaskType() const override { return TodoTaskType::kBoxing; }

 private:
};

class InBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InBoxingTaskNode);
  InBoxingTaskNode() = default;
  ~InBoxingTaskNode() = default;

 private:
};

class OutBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutBoxingTaskNode);
  OutBoxingTaskNode() = default;
  ~OutBoxingTaskNode() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_
