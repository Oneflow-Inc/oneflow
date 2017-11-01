#include "oneflow/core/graph/boxing_task_node.h"

namespace oneflow {

void BoxingTaskNode::Init(
    int64_t machine_id, std::function<void(BoxingOpConf*)> BoxingOpConfSetter) {
  set_machine_id(machine_id);
  set_thrd_loc_id(IDMgr::Singleton()->BoxingThrdLocId());
}

}  // namespace oneflow
