#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/embedding_lookup_model_diff_accumulate_compute_task_node.h"

namespace oneflow {

void EmbeddingLookupMdDiffAccCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto acc_regst = ProduceRegst("acc");
  SoleOutEdge()->AddRegst("acc", acc_regst);
}

void EmbeddingLookupMdDiffAccCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("one", SoleInEdge()->GetSoleRegst());
}

void EmbeddingLookupMdDiffAccCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> one_regst = GetConsumedRegst("one");
  std::shared_ptr<RegstDesc> acc_regst = GetProducedRegst("acc");
  int max_acc_cnt = JobDesc::Singleton()->NumOfPiecesInBatch();
  acc_regst->CopyBlobDescFrom(one_regst.get());
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  BlobDesc* acc_ids_bd = acc_regst->MutBlobDesc(op->Lbn4BnInOp("acc_ids"));
  acc_ids_bd->mut_shape().Set(0, acc_ids_bd->shape().At(0) * max_acc_cnt);
  BlobDesc* acc_val_bd = acc_regst->MutBlobDesc(op->Lbn4BnInOp("acc_val"));
  acc_val_bd->mut_shape().Set(0, acc_val_bd->shape().At(0) * max_acc_cnt);
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnInOpAndRegst(op->input_bns().at(0), one_regst);
  exec_node->BindBnInOpAndRegst(op->input_bns().at(1), one_regst);
  exec_node->BindBnInOpAndRegst(op->output_bns().at(0), acc_regst);
  exec_node->BindBnInOpAndRegst(op->output_bns().at(1), acc_regst);
}

}  // namespace oneflow
