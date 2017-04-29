#include "graph/copy_hd_task_node.h"
#include "operator/operator_manager.h"
#include "operator/copy_op.h"

namespace oneflow {

void CopyHDTaskNode::SetFwInCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = true;
}

void CopyHDTaskNode::SetFwOutCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = false;
}

void CopyHDTaskNode::InitWithFwNode(TaskNode* fw_node) {
  TaskNode::InitWithFwNode(fw_node);
  is_fw_in_copy_ = of_dynamic_cast<CopyHDTaskNode*>(fw_node)->is_fw_in_copy_;
}

void CopyHDTaskNode::FwBuildExecAndEnrollLbn2Regsts(TaskGraph*) {
  return CopyHdBuildExecAndEnrollLbn2Regsts();
}

void CopyHDTaskNode::FwInferShape4LbnInProducedRegsts(TaskGraph*) {
  return CopyHdInferShape4LbnInProducedRegsts();
}

void CopyHDTaskNode::BpBuildExecAndEnrollLbn2Regsts(TaskGraph*) {
  return CopyHdBuildExecAndEnrollLbn2Regsts();
}

void CopyHDTaskNode::BpInferShape4LbnInProducedRegsts(TaskGraph*) {
  return CopyHdInferShape4LbnInProducedRegsts();
}

void CopyHDTaskNode::CopyHdBuildExecAndEnrollLbn2Regsts(){
  auto out_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  BindProducedRegstAndOutEdge(out_regst.get(), SoleOutEdge());
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  out_regst->CopyLbnFrom(in_regst);

  OperatorConf op_conf;
  op_conf.set_name("copy_hd_" + NewUniqueId());
  CopyHdOpConf* copy_hd_conf = op_conf.mutable_copy_hd_conf();
  copy_hd_conf->set_type(IsH2D() ? CopyHdOpConf::H2D : CopyHdOpConf::D2H);

  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = OpMgr::Singleton().ConstructOp(op_conf);
  
  node->BindBnInOpAndRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnInOpAndRegst(node->op()->SoleObn(), out_regst.get());
  
  mut_exec_gph().UpdateSourceAndSink();
  EnrollProducedRegstDesc("out", std::move(out_regst));
}

void CopyHDTaskNode::CopyHdInferShape4LbnInProducedRegsts() {
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  RegstDesc* out_regst = GetRelatedRegst(SoleOutEdge());
  out_regst->CopyShapeFrom(in_regst);
}

} // namespace oneflow
