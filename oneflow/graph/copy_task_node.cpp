#include "graph/copy_task_node.h"
#include "operator/copy_hd_op.h"
#include "operator/comm_net_op.h"
#include "operator/operator_manager.h"

namespace oneflow {

void CopyTaskNode::BuildExecAndEnrollLbn2Regsts(TaskGraph*){
  auto out_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  BindProducedRegstAndOutEdge(out_regst.get(), SoleOutEdge());
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  out_regst->CopyLbnFrom(in_regst);
  
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = ConstructOp();
  
  if (IsFwNode()) {
    node->BindBnInOpAndRegst(node->op()->SoleIbn(), in_regst);
    node->BindBnInOpAndRegst(node->op()->SoleObn(), out_regst.get());
  } else {
    node->BindBnInOpAndRegst(node->op()->SoleOdbn(), in_regst);
    node->BindBnInOpAndRegst(node->op()->SoleIdbn(), out_regst.get());
  }
  
  mut_exec_gph().UpdateSourceAndSink();
  EnrollProducedRegstDesc("copy", std::move(out_regst));
}

void CopyTaskNode::InferShape4LbnInProducedRegsts(TaskGraph*) {
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  RegstDesc* out_regst = GetRelatedRegst(SoleOutEdge());
  out_regst->CopyShapeFrom(in_regst);
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

void CommNetTaskNode::SetFwSender() {
  CHECK(IsFwNode());
  is_fw_sender_ = true;
}

void CommNetTaskNode::SetFwReceiver() {
  CHECK(IsFwNode());
  is_fw_sender_ = false;
}

std::shared_ptr<const Operator> CommNetTaskNode::ConstructOp() const {
  OperatorConf op_conf;
  op_conf.set_name("comm_net_" + NewUniqueId());
  CommNetOpConf* comm_net_conf = op_conf.mutable_comm_net_conf();
  comm_net_conf->set_type(
      IsSender() ? CommNetOpConf::kSender : CommNetOpConf::kReceiver);
  return OpMgr::Singleton().ConstructOp(op_conf);
}

} // namespace oneflow
