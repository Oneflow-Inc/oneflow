#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

void CopyTaskNode::ProduceAllRegstsAndBindEdges() {
  std::string name("copy_out");
  auto out_regst = ProduceRegst(name, 1, kMaxRegisterNum);
  SoleOutEdge()->AddRegst(name, out_regst);
}

void CopyTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("copy_in", SoleInEdge()->GetSoleRegst());
}
void CopyTaskNode::BuildExecGphAndRegst() {
  auto out_regst = GetProducedRegst("copy_out");
  auto in_regst = GetConsumedRegst("copy_in");
  out_regst->CopyBlobDescFrom(in_regst.get());
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = ConstructOp(NewCopyOpConf());
  node->BindBnInOpAndRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnInOpAndRegst(node->op()->SoleObn(), out_regst);
}

void CopyHdTaskNode::Init(const CompTaskNode* comp_task,
                          CopyHdOpConf::Type copy_type) {
  set_machine_id(comp_task->machine_id());
  set_thrd_id(comp_task->thrd_id());
  copy_type_ = copy_type;
}

OperatorConf CopyHdTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_name("copy_hd_" + NewUniqueId());
  conf.mutable_copy_hd_conf()->set_type(copy_type_);
  return conf;
}

void CopyCommNetTaskNode::Init(int64_t machine_id) {
  set_machine_id(machine_id);
  set_thrd_id(IDMgr::Singleton()->CommNetThrdId());
}

OperatorConf CopyCommNetTaskNode::NewCopyOpConf() {
  OperatorConf conf;
  conf.set_name("copy_comm_net_" + NewUniqueId());
  conf.mutable_copy_comm_net_conf();
  return conf;
}

}  // namespace oneflow
