#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void DecodeCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out");
  for (TaskEdge* edge : out_edges()) { edge->AddRegst("out", out_regst); }
}

void DecodeCompTaskNode::ConsumeAllRegsts() {}

void DecodeCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  FOR_RANGE(size_t, i, 0, chain_node()->op_vec().size()) {
    ExecNode* node = mut_exec_gph().NewNode();
    node->mut_op() = chain_node()->op_vec().at(i);
    for (const std::string& obn : node->op()->output_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(obn);
      out_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(obn, out_regst);
    }
    node->InferBlobDescs(parallel_ctx(), device_type());
  }
}

}  // namespace oneflow
