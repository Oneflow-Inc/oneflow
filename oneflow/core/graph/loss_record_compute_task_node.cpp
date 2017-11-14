#include "oneflow/core/graph/loss_record_compute_task_node.h"

namespace oneflow {

void LossRecordCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void LossRecordCompTaskNode::FixThrdLocId() {
  set_thrd_loc_id(IDMgr::Singleton()->PersistenceThrdLocId());
}

}  // namespace oneflow
