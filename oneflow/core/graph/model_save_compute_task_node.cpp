#include "oneflow/core/graph/model_save_compute_task_node.h"

namespace oneflow {

void MdSaveCompTaskNode::NewAllProducedRegst() {}

void MdSaveCompTaskNode::FixThrdLocId() {
  set_thrd_loc_id(IDMgr::Singleton()->PersistenceThrdLocId());
}

}  // namespace oneflow
