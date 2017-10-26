#include "oneflow/core/graph/model_update_compute_task_node.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

void MdUpdtCompTaskNode::NewAllProducedRegst() {
  NewProducedRegst("model_tmp", 1, 1);
  NewProducedRegst("model", 3, kMaxRegisterNum);
}

}  // namespace oneflow
