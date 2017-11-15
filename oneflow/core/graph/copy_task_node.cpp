#include "oneflow/core/graph/copy_task_node.h"

namespace oneflow {

void CopyTaskNode::ProduceAllRegstsAndBindEdges() {
  std::string name("copy_out");
  auto out_regst = ProduceRegst(name, 1, kMaxRegisterNum);
  SoleOutEdge()->AddRegst(name, out_regst);
}

void CopyTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("copy_in", SoleInEdge()->GetSoleRegst());
}
void CopyTaskNode::BuildRegsts() {
  auto out_regst = GetProducedRegst("copy_out");
  auto in_regst = GetConsumedRegst("copy_in");
  out_regst->CopyBlobDescFrom(in_regst.get());
}

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
