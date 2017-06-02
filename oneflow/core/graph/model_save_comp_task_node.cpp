#include "oneflow/core/graph/model_save_comp_task_node.h"
#include "oneflow/core/graph/model_save_task_graph.h"

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

    OperatorConf op_conf;
    op_conf.set_name("model_save_op" + updt_task->node_id_str());
    op_conf.mutable_model_save_conf();
    for (const auto& pair : GetRelatedRegst(SoleInEdge())->lbn2shape()) {
      op_conf.mutable_model_save_conf()->add_lbns(pair.first);
    }

    ExecNode* exec_node = mut_exec_gph().NewNode();
    exec_node->mut_op() = OpMgr::Singleton().ConstructOp(op_conf);
    for (const std::string& ibn : exec_node->op()->input_bns()) {
      exec_node->BindBnInOpAndRegst(ibn, GetRelatedRegst(SoleInEdge()));
    }
    mut_exec_gph().UpdateSourceAndSink();
  } else {
    UNEXPECTED_RUN();
  }
}

void MdSaveCompTaskNode::InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) {
  CHECK(IsFwNode());
}

} // namespace oneflow
