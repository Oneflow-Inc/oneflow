#include "oneflow/core/graph/reduce_split_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceSplitCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    while (dst_node->GetTaskType() != TaskType::kNormalMdUpdt) {
      dst_node = dst_node->SoleOutEdge()->dst_node();
    }
    CompTaskNode* mdupdt_node = dynamic_cast<CompTaskNode*>(dst_node);
    std::string out_regst_name = "out_" + std::to_string(mdupdt_node->reduce_id());
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name, false, 1, 1);
    edge->AddRegst(out_regst_name, out_regst);
  }
}

void ReduceSplitCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", this->SoleInEdge()->GetSoleRegst());
}

void ReduceSplitCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_split_op = this->logical_node()->SoleOp();
  node->mut_op() = reduce_split_op;
  node->BindBnWithRegst(reduce_split_op->SoleIbn(), GetSoleConsumedRegst("in"));

  FOR_RANGE(size_t, i, 0, reduce_split_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    CHECK(out_regst.get() != nullptr);
    const std::string& obn = reduce_split_op->output_bns().Get(i);
    out_regst->AddLbi(reduce_split_op->BnInOp2Lbi(obn));
    node->BindBnWithRegst(obn, out_regst);
  }
  // TODO(jiyuan): copy blob desc from bw or fw node
  // node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
