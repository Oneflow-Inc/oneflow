#include "oneflow/core/graph/accuracy_accumulate_compute_task_node.h"

namespace oneflow {

void AccuracyAccCompTaskNode::InferProducedDataRegstTimeShape() {
  int64_t accuracy_acc_output_num = Global<JobDesc>::Get()->TotalBatchNum()
                                    * Global<JobDesc>::Get()->NumOfPiecesInBatch()
                                    / Global<JobDesc>::Get()->PieceNumOfPrintAccuracy();
  std::shared_ptr<Shape> time_shape(new Shape({accuracy_acc_output_num}));
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
