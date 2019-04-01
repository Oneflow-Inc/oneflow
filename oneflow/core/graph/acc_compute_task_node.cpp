#include "oneflow/core/graph/acc_compute_task_node.h"
#include "oneflow/core/operator/acc_op.h"

namespace oneflow {

void AccCompTaskNode::InferProducedDataRegstTimeShape() {
  auto TimeShape4Ibn = [&](const std::string& ibn) -> const Shape* {
    return GetSoleConsumedRegst("one")->data_regst_time_shape().get();
  };
  std::shared_ptr<Shape> time_shape(new Shape());
  logical_node()->SoleOp()->InferOutputBlobTimeShape(TimeShape4Ibn, parallel_ctx(),
                                                     time_shape.get());
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
