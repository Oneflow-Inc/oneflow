#include "oneflow/core/graph/model_update_compute_task_node.h"
#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/model_save_compute_task_node.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

void MdUpdtCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto model_regst = ProduceRegst("model_tmp", 1, 1);
  auto model_tmp_regst = ProduceRegst("model", 3, kMaxRegisterNum);
  for (TaskEdge* out_edge : out_edges()) {
    TaskNode* src_node = out_edge->src_node();
    if (dynamic_cast<ForwardCompTaskNode*>(src_node) != nullptr) {
      out_edge->AddRegst("model", model_regst);
      out_edge->AddRegst("model_tmp", model_tmp_regst);
    } else if (dynamic_cast<BackwardCompTaskNode*>(src_node) != nullptr
               || dynamic_cast<MdSaveCompTaskNode*>(src_node) != nullptr) {
      out_edge->AddRegst("model", model_regst);
    } else {
      UNEXPECTED_RUN();
    }
  }
}

void MdUpdtCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("model_diff_acc", SoleInEdge()->GetSoleRegst());
}

void MdUpdtCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  auto model_diff_acc_regst = SoleInEdge()->GetSoleRegst();
  node->BindBnInOpAndRegst("model_diffs", model_diff_acc_regst);
  auto model_regst = GetProducedRegst("model");
  node->BindBnInOpAndRegst(node->op()->SoleObn(), model_regst);
  auto data_tmp_regst = ProduceRegst("data_tmp", 1, 1);
  for (const std::string dtbn : node->op()->data_tmp_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
    data_tmp_regst->AddLbn(lbn);
    node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
  }
}

void MdUpdtCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  task_proto->set_random_seed(random_seed_);
}

}  // namespace oneflow
