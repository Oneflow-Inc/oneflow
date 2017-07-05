#include "oneflow/core/graph/model_diff_accumulate_comp_task_node.h"
#include "oneflow/core/graph/model_diff_accumulate_task_graph.h"

namespace oneflow {

void MdDiffAccCompTaskNode::BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) {
  CHECK(IsFwNode());
  auto md_diff_acc_gph = static_cast<MdDiffAccTaskGraph*>(gph);
  fw_task_ = md_diff_acc_gph->GetFwTaskFromParallelId(parallel_id());
  TaskNode* bp_task = fw_task_->GetBpNode();
  std::shared_ptr<RegstDesc> model_diff_regst =
      bp_task->GetProducedRegstDesc("model_diff");
  // faker task node
  if (chain_node()->op_vec().empty()) {
    BindProducedRegstAndOutEdge(model_diff_regst, SoleOutEdge());
    return;
  }
  // comp task node
  NewProducedRegstDesc("model_diff_acc");
  auto model_diff_acc_regst = GetProducedRegstDesc("model_diff_acc");

  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = chain_node()->SoleOp();
  const std::string ibn = "model_diff";
  if (in_edges().empty()) {
    exec_node->BindBnInOpAndRegst(ibn, model_diff_regst);
    SubscribeRegstDesc(ibn, model_diff_regst);
  } else {
    exec_node->BindBnInOpAndRegst(ibn, GetRelatedRegst(SoleInEdge()));
    SubscribeRegstDesc(ibn, GetRelatedRegst(SoleInEdge()));
  }
  model_diff_acc_regst->CopyLbnFrom(GetSubscribedRegstDesc("model_diff").get());
  exec_node->BindBnInOpAndRegst(exec_node->op()->SoleObn(),
                                model_diff_acc_regst);
  mut_exec_gph().UpdateSourceAndSink();
}

void MdDiffAccCompTaskNode::InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) {
  CHECK(IsFwNode());
  if (!chain_node()->op_vec().empty()) {
    std::shared_ptr<RegstDesc> in_regst = GetSubscribedRegstDesc("model_diff");
    std::shared_ptr<RegstDesc> out_regst =
        GetProducedRegstDesc("model_diff_acc");
    out_regst->CopyShapeFrom(in_regst.get());
  }
}

}  // namespace oneflow
