#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/operator/copy_hd_op.h"
#include "oneflow/core/operator/copy_comm_net_op.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

void CopyTaskNode::BuildExecAndEnrollLbn2Regsts(TaskGraph*){
  auto out_regst = NewProducedRegstDesc("copy_out");
  BindProducedRegstAndOutEdge(out_regst, SoleOutEdge());
  std::shared_ptr<RegstDesc> in_regst = GetRelatedRegst(SoleInEdge());
  SubscribeRegstDesc("copy_in", in_regst);
  out_regst->CopyLbnFrom(in_regst.get());
  
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = ConstructOp();
  
  if (IsFwNode()) {
    node->BindBnInOpAndRegst(node->op()->SoleIbn(), in_regst);
    node->BindBnInOpAndRegst(node->op()->SoleObn(), out_regst);
  } else {
    node->BindBnInOpAndRegst(node->op()->SoleOdbn(), in_regst);
    node->BindBnInOpAndRegst(node->op()->SoleIdbn(), out_regst);
  }
  
  mut_exec_gph().UpdateSourceAndSink();
}

void CopyTaskNode::InferShapeOfBlobsInProducedRegsts(TaskGraph*) {
  std::shared_ptr<RegstDesc> in_regst = GetRelatedRegst(SoleInEdge());
  std::shared_ptr<RegstDesc> out_regst = GetRelatedRegst(SoleOutEdge());
  out_regst->CopyShapeFrom(in_regst.get());
}

void CopyHDTaskNode::SetFwInCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = true;
}

void CopyHDTaskNode::SetFwOutCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = false;
}

std::shared_ptr<const Operator> CopyHDTaskNode::ConstructOp() const {
  OperatorConf op_conf;
  op_conf.set_name("copy_hd_" + NewUniqueId());
  CopyHdOpConf* copy_hd_conf = op_conf.mutable_copy_hd_conf();
  copy_hd_conf->set_type(IsH2D() ? CopyHdOpConf::H2D : CopyHdOpConf::D2H);
  return OpMgr::Singleton().ConstructOp(op_conf);
}

std::shared_ptr<const Operator> CopyCommNetTaskNode::ConstructOp() const {
  OperatorConf op_conf;
  op_conf.set_name("comm_net_" + NewUniqueId());
  op_conf.mutable_copy_comm_net_conf();
  return OpMgr::Singleton().ConstructOp(op_conf);
}

} // namespace oneflow
