#include "oneflow/core/graph/nccl_reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclReduceScatterCompTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
}

void NcclReduceScatterCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void NcclReduceScatterCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> nccl_reduce_scatter_op = this->logical_node()->SoleOp();
  node->mut_op() = nccl_reduce_scatter_op;
  node->BindBnWithRegst(nccl_reduce_scatter_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(nccl_reduce_scatter_op->BnInOp2Lbi(nccl_reduce_scatter_op->SoleObn()));
  node->BindBnWithRegst(nccl_reduce_scatter_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
