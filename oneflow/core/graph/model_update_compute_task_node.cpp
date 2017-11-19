#include "oneflow/core/graph/model_update_compute_task_node.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

void MdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("model_tmp", 1, 1);
  ProduceRegst("model", 3, kMaxRegisterNum);
  auto model_regst = GetProduceRegst("model");
  auto model_tmp_regst = GetProduceRegst("model_tmp");
  for (TaskEdge* out_edge : out_edges()) {
    TaskNode* src_node = out_edge->src_node();
    if (dynamic_cast<ForwardChainNode*>(src_node)) {
      out_edge->AddRegst("model", model_regst);
      out_edge->AddRegst("model_tmp", model_tmp_regst);
    } else if (dynamic_cast<BackwardChainNode*>(src_node)
               || dynamic_cast<MdSaveChainNode*>(src_node)) {
      out_edge->AddRegst("model", model_regst);
    } else {
      UNEXPECTED_RUN();
    }
  }
}

void MdUpdtCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("model_diff_acc", SoleInEdge()->GetSoleRegst());
}

bool MdUpdtCompTaskNode::IsReadyForBuild() {
  return GetProducedRegst("model")->IsLocked();
}

void MdUpdtCompTaskNode::Build() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  auto model_diff_acc_regst = SoleInEdge->GetSoleRegst();
  node->BindBnInOpAndRegst("model_diffs", model_diff_acc_regst);
  auto model_regst = GetProducedRegst("model");
  node->BindBnInOpAndRegst(exec->op()->SoleObn(), model_regst);
  ProduceRegst("data_tmp", 1, 1);
  auto data_tmp_regst = GetProducedRegst("data_tmp");
  for (const std::string dtbn : node->op()->data_tmp_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
    data_tmp_regst->EnrollDataTmpBn(lbn);
    node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
  }
}

void MdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  task_proto->set_random_seed(random_seed_);
}

}  // namespace oneflow
