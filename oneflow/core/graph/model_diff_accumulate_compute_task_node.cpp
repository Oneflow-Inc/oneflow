#include "oneflow/core/graph/model_diff_accumulate_compute_task_node.h"

namespace oneflow {

void MdDiffAccCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<Shape> time_shape(new Shape({GlobalJobDesc().TotalBatchNum()}));
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
