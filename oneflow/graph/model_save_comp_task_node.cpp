#include "graph/model_save_comp_task_node.h"
#include "graph/model_save_task_graph.h"

namespace oneflow {

void MdSaveCompTaskNode::BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) {
  CHECK(IsFwNode());
  auto md_save_gph = of_dynamic_cast<MdSaveTaskGraph*> (gph);
  CompTaskNode* updt_task = md_save_gph->update_task();
  if (in_edges().empty()) {
    BindProducedRegstAndOutEdge(updt_task->GetProducedRegstDesc("model"),
                                SoleOutEdge());
  } else if (out_edges().empty()) {
    SubscribeRegstDesc("model", GetRelatedRegst(SoleInEdge()));
  } else {
    UNEXPECTED_RUN();
  }
}

void MdSaveCompTaskNode::InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) {
  CHECK(IsFwNode());
}

} // namespace oneflow
