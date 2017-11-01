#include "oneflow/core/graph/source_compute_task_node.h"

namespace oneflow {

void SourceCompTaskNode::ProduceAllRegstsAndBindEdges() {
  NewProducedRegst("out", 1, kMaxRegisterNum);
}

void SourceCompTaskNode::FixThrdLocId() {
  set_thrd_loc_id(IDMgr::Singleton()->PersistenceThrdLocId());
}

}  // namespace oneflow
