#include "oneflow/core/graph/source_compute_task_node.h"

namespace oneflow {

void SourceCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", 1, kMaxRegisterNum);
}

void SourceCompTaskNode::FixThrdId() {
  set_thrd_id(IDMgr::Singleton()->AllocatePersistenceThrdId(machine_id()));
}

}  // namespace oneflow
