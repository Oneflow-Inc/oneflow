#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"

namespace oneflow {

void LossAccCompTaskNode::InferProducedRegstTimeShape() {
  std::shared_ptr<Shape> time_shape = std::make_shared<Shape>(std::vector<int64_t>(
      {Global<JobDesc>::Get()->TotalBatchNum() * Global<JobDesc>::Get()->NumOfPiecesInBatch()
       / Global<JobDesc>::Get()->PieceNumOfPrintLoss()}));
  for (auto& pair : produced_regsts()) { pair.second->mut_time_shape() = time_shape; }
}

}  // namespace oneflow
