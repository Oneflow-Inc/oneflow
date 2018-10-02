#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"

namespace oneflow {

void LossAccCompTaskNode::InferProducedDataRegstTimeShape() {
  int64_t loss_acc_output_num = Global<JobDesc>::Get()->TotalBatchNum()
                                * Global<JobDesc>::Get()->NumOfPiecesInBatch()
                                / Global<JobDesc>::Get()->PieceNumOfPrintLoss();
  std::shared_ptr<Shape> time_shape(new Shape({loss_acc_output_num}));
  ForEachProducedDataRegst([&time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
