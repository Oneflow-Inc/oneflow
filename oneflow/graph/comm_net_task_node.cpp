#include "graph/comm_net_task_node.h"
#include "operator/operator_manager.h"
#include "operator/copy_op.h"

namespace oneflow {

void CommNetTaskNode::CommNetBuildExecAndEnrollLbn2Regsts() {
  auto out_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  BindProducedRegstAndOutEdge(out_regst.get(), SoleOutEdge());
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  out_regst->CopyLbnFrom(in_regst);

  OperatorConf op_conf;
  op_conf.set_name("comm_net_" + NewUniqueId());
  CommNetOpConf* comm_net_conf = op_conf.mutable_comm_net_conf();
  comm_net_conf->set_comm_net_type(
      IsSender() ? CommNetOpConf::SENDER : CommNetOpConf::RECEIVER);
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = OpMgr::Singleton().ConstructOp(op_conf);

  node->BindBnInOpAndRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnInOpAndRegst(node->op()->SoleObn(), out_regst.get());
  
  mut_exec_gph().UpdateSourceAndSink();
  EnrollProducedRegstDesc("out", std::move(out_regst));
}

void CommNetTaskNode::CommNetInferShape4LbnInProducedRegsts() {
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  RegstDesc* out_regst = GetRelatedRegst(SoleOutEdge());
  out_regst->CopyShapeFrom(in_regst);
}

void FwBuildExecAndEnrollLbn2Regsts(TaskGraph*) override {
  return CommNetBuildExecAndEnrollLbn2Regsts();
}

void FwInferShape4LbnInProducedRegsts(TaskGraph*) override {
  return CommNetInferShape4LbnInProducedRegsts();
}

void BpBuildExecAndEnrollLbn2Regsts(TaskGraph*) override {
  return CommNetBuildExecAndEnrollLbn2Regsts();
}

void BpInferShape4LbnInProducedRegsts(TaskGraph*) override {
  return CommNetInferShape4LbnInProducedRegsts();
}

} // namespace oneflow
