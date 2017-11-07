#include "oneflow/core/graph/model_update_compute_task_node.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

void MdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("model_tmp", 1, 1);
  ProduceRegst("model", 3, kMaxRegisterNum);
}

}  // namespace oneflow
