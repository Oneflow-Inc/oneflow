#ifndef ONEFLOW_CORE_GRAPH_ACCURACY_PRINT_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_ACCURACY_PRINT_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

class AccuracyPrintCompTaskNode final : public SinkCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccuracyPrintCompTaskNode);
  AccuracyPrintCompTaskNode() = default;
  ~AccuracyPrintCompTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kAccuracyPrint; }
  bool MayBeBlocked() const override { return true; }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACCURACY_PRINT_COMPUTE_TASK_NODE_H_
