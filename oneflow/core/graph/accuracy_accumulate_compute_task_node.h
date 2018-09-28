#ifndef ONEFLOW_CORE_GRAPH_ACCURACY_ACCUMULATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_ACCURACY_ACCUMULATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/accumulate_compute_task_node.h"

namespace oneflow {

class AccuracyAccCompTaskNode final : public AccCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccuracyAccCompTaskNode);
  AccuracyAccCompTaskNode() = default;
  ~AccuracyAccCompTaskNode() override = default;
  TaskType GetTaskType() const override { return TaskType::kAccuracyAcc; }

 private:
  void InferProducedRegstTimeShape() override {
    std::shared_ptr<Shape> time_shape;
    time_shape.reset(new Shape({Global<JobDesc>::Get()->TotalBatchNum()
                                * Global<JobDesc>::Get()->NumOfPiecesInBatch()
                                / Global<JobDesc>::Get()->PieceNumOfPrintAccuracy()}));
    for (auto& pair : produced_regsts()) { pair.second->mut_time_shape() = time_shape; }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_ACCURACY_ACCUMULATE_COMPUTE_TASK_NODE_H_
