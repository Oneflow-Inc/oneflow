#include "oneflow/core/graph/nccl_all_reduce_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclAllReduceCompTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
}

void NcclAllReduceCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void NcclAllReduceCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> nccl_all_reduce_op = this->logical_node()->SoleOp();
  node->mut_op() = nccl_all_reduce_op;
  node->BindBnWithRegst(nccl_all_reduce_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(nccl_all_reduce_op->BnInOp2Lbi(nccl_all_reduce_op->SoleObn()));
  node->BindBnWithRegst(nccl_all_reduce_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
