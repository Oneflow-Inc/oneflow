#include "oneflow/core/graph/source_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void SourceCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("data_tmp", 1, 1);
  auto out_regst = ProduceRegst("out", 1, kMaxRegisterNum);
  SoleOutEdge()->AddRegst("out", out_regst);
}

void SourceCompTaskNode::ConsumeAllRegsts() {}

void SourceCompTaskNode::BuildExecGphAndRegst() {
  auto out_regst = GetProducedRegst("out");
  auto data_tmp_regst = GetProducedRegst("data_tmp");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  auto& data_output_lbns = chain_node()->data_output_lbns();
  for (const std::string& obn : node->op()->output_bns()) {
    auto& lbn = node->op()->Lbn4BnInOp(obn);
    if (data_output_lbns.find(lbn) == data_output_lbns.end()) {
      data_tmp_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(obn, data_tmp_regst);
    } else {
      out_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(obn, out_regst);
    }
  }
  node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx());
}

void SourceCompTaskNode::FixThrdId() {
  set_thrd_id(IDMgr::Singleton()->AllocatePersistenceThrdId(machine_id()));
}

}  // namespace oneflow
