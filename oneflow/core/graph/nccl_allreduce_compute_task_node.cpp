#include "oneflow/core/graph/nccl_allreduce_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclAllreduceCompTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
}

void NcclAllreduceCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void NcclAllreduceCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> nccl_allreduce_op = this->logical_node()->SoleOp();
  node->mut_op() = nccl_allreduce_op;
  node->BindBnWithRegst(nccl_allreduce_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(nccl_allreduce_op->BnInOp2Lbi(nccl_allreduce_op->SoleObn()));
  node->BindBnWithRegst(nccl_allreduce_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
