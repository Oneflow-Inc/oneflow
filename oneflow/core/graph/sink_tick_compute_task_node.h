#ifndef ONEFLOW_CORE_GRAPH_SINK_TICK_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_SINK_TICK_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

class SinkTickCompTaskNode final : public SinkCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SinkTickCompTaskNode);
  SinkTickCompTaskNode() = default;
  ~SinkTickCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kSinkTick; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SINK_TICK_COMPUTE_TASK_NODE_H_
