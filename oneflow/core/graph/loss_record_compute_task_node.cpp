#include "oneflow/core/graph/loss_record_compute_task_node.h"

namespace oneflow {

void LossRecordCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void LossRecordCompTaskNode::FixThrdId() {
  set_thrd_id(IDMgr::Singleton()->AllocatePersistenceThrdId(machine_id()));
}

}  // namespace oneflow
