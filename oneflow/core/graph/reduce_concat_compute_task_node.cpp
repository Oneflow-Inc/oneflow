#include "oneflow/core/graph/reduce_concat_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceConcatCompTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
}

void ReduceConcatCompTaskNode::ConsumeAllRegsts() {
  struct EdgeInfo {
    int64_t bw_node_order;
    TaskEdge* edge;
  };
  std::vector<EdgeInfo> edge_infos;
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (src_node->GetTaskType() != TaskType::kNormalBackward) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    CompTaskNode* bw_node = dynamic_cast<CompTaskNode*>(src_node);
    EdgeInfo edge_info{bw_node->order_in_graph(), edge};
    edge_infos.emplace_back(edge_info);
  }
  std::sort(edge_infos.begin(), edge_infos.end(), [](const EdgeInfo& lhs, const EdgeInfo& rhs) {
    return lhs.bw_node_order < rhs.bw_node_order;
  });
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

}  // namespace oneflow
