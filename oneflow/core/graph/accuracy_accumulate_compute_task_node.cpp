#include "oneflow/core/graph/accuracy_accumulate_compute_task_node.h"

namespace oneflow {

void AccuracyAccCompTaskNode::InferProducedDataRegstTimeShape() {
  int64_t accuracy_acc_output_num = GlobalJobDesc().TotalBatchNum()
                                    * GlobalJobDesc().NumOfPiecesInBatch()
                                    / GlobalJobDesc().PieceNumOfPrintAccuracy();
  std::shared_ptr<Shape> time_shape(new Shape({accuracy_acc_output_num}));
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
