#ifndef ONEFLOW_CORE_GRAPH_CALLBACK_NOTIFY_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_CALLBACK_NOTIFY_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

class CallbackNotifyCompTaskNode final : public SinkCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CallbackNotifyCompTaskNode);
  CallbackNotifyCompTaskNode() = default;
  ~CallbackNotifyCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kCallbackNotify; }
  bool IsIndependent() const override { return true; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CALLBACK_NOTIFY_COMPUTE_TASK_NODE_H_
