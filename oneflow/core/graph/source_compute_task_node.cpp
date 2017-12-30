#include "oneflow/core/graph/source_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void SourceCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("data_tmp", 1, 1);
  auto out_regst = ProduceRegst("out");
  SoleOutEdge()->AddRegst("out", out_regst);
}

void SourceCompTaskNode::ConsumeAllRegsts() {}

void SourceCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  const auto& data_output_lbns = chain_node()->data_output_lbns();
  for (const std::string& obn : node->op()->output_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(obn);
    if (data_output_lbns.find(lbn) == data_output_lbns.end()) {
      data_tmp_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(obn, data_tmp_regst);
    } else {
      out_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(obn, out_regst);
    }
  }
  for (const std::string& dtbn : node->op()->data_tmp_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
    data_tmp_regst->AddLbn(lbn);
    node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
  }
  node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx());
}

void SourceCompTaskNode::FixThrdId() {
  set_thrd_id(IDMgr::Singleton()->AllocatePersistenceThrdId(machine_id()));
}

}  // namespace oneflow
