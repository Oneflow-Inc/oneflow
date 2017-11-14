#include "oneflow/core/graph/copy_task_node.h"

namespace oneflow {

void CopyHdTaskNode::Init(const CompTaskNode* comp_task,
                          CopyHdOpConf::Type copy_type) {
  set_machine_id(comp_task->machine_id());
  set_thrd_loc_id(comp_task->thrd_loc_id());
}

void CopyCommNetTaskNode::Init(int64_t machine_id) {
  set_machine_id(machine_id);
  set_thrd_loc_id(IDMgr::Singleton()->CommNetThrdLocId());
}

}  // namespace oneflow
