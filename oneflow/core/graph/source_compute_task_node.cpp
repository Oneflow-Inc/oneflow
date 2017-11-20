#include "oneflow/core/graph/source_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void SourceCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto out_regst = ProduceRegst("out", 1, kMaxRegisterNum);
  for (TaskEdge* edge : out_edges()) { edge->AddRegst("out", out_regst); }
}

void SourceCompTaskNode::ConsumeAllRegsts() {}

void SourceCompTaskNode::BuildExecGphAndRegst() {
  auto out_regst = GetProducedRegst("out");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  for (const std::string& obn : node->op()->output_bns()) {
    node->BindBnInOpAndRegst(obn, out_regst);
  }
  node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx());
}

void SourceCompTaskNode::FixThrdId() {
  set_thrd_id(IDMgr::Singleton()->AllocatePersistenceThrdId(machine_id()));
}

}  // namespace oneflow
