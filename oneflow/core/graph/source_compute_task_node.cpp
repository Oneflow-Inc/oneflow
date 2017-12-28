#include "oneflow/core/graph/source_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void SourceCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("activation");
  auto out_regst = ProduceRegst("out");
  SoleOutEdge()->AddRegst("out", out_regst);
}

void SourceCompTaskNode::ConsumeAllRegsts() {}

void SourceCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> activation_regst = GetProducedRegst("activation");
  activation_regst->set_register_num_range(1, 1);
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  const auto& data_output_lbns = chain_node()->data_output_lbns();
  for (const std::string& obn : node->op()->output_bns()) {
    const std::string& lbn = node->op()->Lbn4BnInOp(obn);
    if (data_output_lbns.find(lbn) == data_output_lbns.end()) {
      activation_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(obn, activation_regst);
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
