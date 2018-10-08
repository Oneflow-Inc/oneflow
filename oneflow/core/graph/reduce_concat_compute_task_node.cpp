#include "oneflow/core/graph/reduce_concat_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceConcatCompTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
}

void ReduceConcatCompTaskNode::ConsumeAllRegsts() {
  std::vector<EdgeInfo> edge_infos;
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (src_node->GetTaskType() != TaskType::kNormalBackward) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    CompTaskNode* bw_node = dynamic_cast<CompTaskNode*>(src_node);
    EdgeInfo edge_info{edge, bw_node->order_in_graph()};
    edge_infos.emplace_back(edge_info);
  }
  SortEdges(&edge_infos);
  FOR_RANGE(size_t, idx, 0, edge_infos.size()) {
    ConsumeRegst("in_" + std::to_string(idx), edge_infos[idx].edge->GetSoleRegst());
  }
}

void ReduceConcatCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_concat_op = this->logical_node()->SoleOp();
  node->mut_op() = reduce_concat_op;
  FOR_RANGE(size_t, i, 0, reduce_concat_op->input_bns().size()) {
    node->BindBnWithRegst(reduce_concat_op->input_bns().Get(i),
                          GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(reduce_concat_op->BnInOp2Lbi(reduce_concat_op->SoleObn()));
  node->BindBnWithRegst(reduce_concat_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void ReduceConcatCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  CHECK_EQ(GetRankCtx().TotalSegmentCount(), 1);
  ctx.EnableMemSharing4Regst(GetProducedRegst("out").get(), 0);

  size_t concat_num = consumed_regsts().size();
  int64_t offset = 0;
  FOR_RANGE(int32_t, idx, 0, concat_num) {
    RegstDesc* concat_in_regst = GetSoleConsumedRegst("in_" + std::to_string(idx)).get();
    ctx.EnableMemSharing4Regst(concat_in_regst, offset);
    offset += InferRegstSize(*concat_in_regst);
  }
}

void ReduceConcatCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
