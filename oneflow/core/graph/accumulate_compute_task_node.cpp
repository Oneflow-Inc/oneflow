#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void AccumulateCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto acc_regst = ProduceRegst("acc", false);
  SoleOutDataEdge()->AddRegst("acc", acc_regst);
}

void AccumulateCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("one", SoleInDataEdge()->GetSoleRegst());
}

void AccumulateCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> one_regst = GetSoleConsumedRegst("one");
  std::shared_ptr<RegstDesc> acc_regst = GetProducedRegst("acc");
  acc_regst->CopyBlobDescFrom(one_regst.get());
  std::shared_ptr<const Operator> op = logical_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnWithRegst(op->SoleIbn(), one_regst);
  exec_node->BindBnWithRegst(op->SoleObn(), acc_regst);
  acc_regst->ForEachLbi([acc_regst](const LogicalBlobId& lbi) {
    const BlobDesc* blob_desc = acc_regst->GetBlobDesc(lbi);
    CHECK_EQ(blob_desc->is_dynamic(), false);
    CHECK_EQ(blob_desc->is_tensor_list(), false);
  });
}

}  // namespace oneflow
