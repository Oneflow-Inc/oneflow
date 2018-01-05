#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void ForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> activation_regst = ProduceRegst("activation");
  std::shared_ptr<RegstDesc> data_tmp_regst = ProduceRegst("data_tmp");
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out");
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (IsBackwardTaskType(dst_node->GetTaskType())) {
      edge->AddRegst("activation", activation_regst);
      edge->AddRegst("data_tmp", data_tmp_regst);
    }
    edge->AddRegst("out", out_regst);
  }
}

void ForwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    if (src_node->GetTaskType() == TaskType::kMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else {
      VirtualConsumeInRegst(edge);
    }
  }
}

void ForwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  BuildActivationRegst();
  BuildModelAndTmpRegsts();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) {
    node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx());
  });
}

void ForwardCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<std::string> found_lbns;
    for (ExecEdge* out_edge : cur_node->out_edges()) {
      CHECK(found_lbns.insert(out_edge->lbn()).second);
    }
    for (const std::string& obn : cur_node->op()->output_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(obn);
      if (found_lbns.find(lbn) != found_lbns.end()) { continue; }
      out_regst->AddLbn(lbn);
      cur_node->BindBnInOpAndRegst(obn, out_regst);
    }
  });
}

void ForwardCompTaskNode::BuildActivationRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetProducedRegst("activation");
  mut_exec_gph().ForEachEdge([&](const ExecEdge* edge) {
    activation_regst->AddLbn(edge->lbn());
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_regst);
  });
}

void ForwardCompTaskNode::BuildModelAndTmpRegsts() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  std::shared_ptr<RegstDesc> model_regst = GetConsumedRegst("model");
  std::shared_ptr<RegstDesc> model_tmp_regst = GetConsumedRegst("model_tmp");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
      data_tmp_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(mtbn);
      model_tmp_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
    for (const std::string& mbn : node->op()->model_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(mbn);
      model_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(mbn, model_regst);
    }
  });
}

void ForwardCompTaskNode::LockRegsts() {
  TaskNode::LockRegsts();
  TryLockConsumedRegst("model");
  TryLockConsumedRegst("model_tmp");
}

void ForwardCompTaskNode::FixRegisterNumRange() {
  int32_t max_seq_size = GetConsumedRegst("in")->MaxSeqSize();
  GetProducedRegst("activation")->set_min_register_num(max_seq_size);
  GetProducedRegst("data_tmp")->set_min_register_num(max_seq_size);
  GetProducedRegst("out")->set_min_register_num(max_seq_size);
}

}  // namespace oneflow
