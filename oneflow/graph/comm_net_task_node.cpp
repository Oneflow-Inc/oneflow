#include "graph/comm_net_task_node.h"
#include "operator/operator_factory.h"
#include "operator/copy_op.h"

namespace oneflow {

void CommNetTaskNode::BuildExecAndProducedRegstsForNetCopy(TaskGraph* gph){
  auto out_regst = of_make_unique<DisContigRegstDesc> ();
  BindProducedRegstAndOutEdge(out_regst.get(), SoleOutEdge());
  EnrollProducedRegstDesc("net_copy", std::move(out_regst));
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  GetProducedRegstDesc("net_copy")->CopyLbn2ShapeMap(in_regst);
  // There no op conf for net task yet
  TODO();
}

void CommNetTaskNode::FwBuildExecAndProducedRegsts(TaskGraph* gph) {
  BuildExecAndProducedRegstsForNetCopy(gph);
}

void CommNetTaskNode::BpBuildExecAndProducedRegsts(TaskGraph* gph) {
  BuildExecAndProducedRegstsForNetCopy(gph);
}

} // namespace oneflow
