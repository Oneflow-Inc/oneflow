#include "graph/comm_net_task_node.h"
#include "operator/operator_factory.h"
#include "operator/copy_op.h"

namespace oneflow {

void CommNetTaskNode::BuildExecAndProducedRegstsForNetCopy(TaskGraph* gph){
  auto out_regst = of_make_unique<DisContigRegstDesc> ();
  BindProducedRegstAndOutEdge(out_regst.get(), SoleOutEdge());
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  out_regst->CopyLbn2ShapeMap(in_regst);

  OperatorConf op_conf;
  op_conf.set_name("comm_net_" + NewUniqueId());
  CommNetOpConf* comm_net_conf = op_conf.mutable_comm_net_conf();
  comm_net_conf->set_comm_net_type(
      IsSender() ? CommNetOpConf::SENDER : CommNetOpConf::RECEIVER);
  ExecNode* node = mut_exec_gph().NewFinalNode();
  node->mut_op() = ConstructOpFromPbConf(op_conf);

  node->BindBnInOpAndRegst(node->op()->SoleIbn(), in_regst);
  node->BindBnInOpAndRegst(node->op()->SoleObn(), out_regst.get());
  
  mut_exec_gph().UpdateSourceAndSink();
  EnrollProducedRegstDesc("comm_net", std::move(out_regst));
}

void CommNetTaskNode::FwBuildExecAndProducedRegsts(TaskGraph* gph) {
  BuildExecAndProducedRegstsForNetCopy(gph);
}

void CommNetTaskNode::BpBuildExecAndProducedRegsts(TaskGraph* gph) {
  BuildExecAndProducedRegstsForNetCopy(gph);
}

} // namespace oneflow
