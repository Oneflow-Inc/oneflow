#include "graph/comm_net_task_node.h"
#include "operator/operator_factory.h"
#include "operator/copy_op.h"

namespace oneflow {

void BuildExecAndProducedRegstsForCopy(TaskGraph* gph) {
  auto out_regst = make_unique<DisContigRegstDesc> ();
  BindProducedRegstAndOutEdge(out_regst.get(), SoleOutEdge());
  EnrollProducedRegstDesc("net_copy", std::move(out_regst));
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  out_regst.CopyLbn2ShapeMap(in_regst);
  // There no op conf for net task yet
  TODO();
}

void CommNetTaskNode::FwBuildExecAndProducedRegsts(TaskGraph* gph) {
  BuildExecAndProducedRegstsFroCopy(gph);
}

void CommNetTaskNode::BpBuildExecAndProducedRegsts(TaskGraph* gph) {
  BuildExecAndProducedRegstsFroCopy(gph);
}

} // namespace oneflow
