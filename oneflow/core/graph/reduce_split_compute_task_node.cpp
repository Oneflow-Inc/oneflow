#include "oneflow/core/graph/reduce_split_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

void ReduceSplitCompTaskNode::ProduceAllRegstsAndBindEdges() {
  struct EdgeInfo {
    int64_t bw_node_order;
    TaskEdge* edge;
  };
  std::vector<EdgeInfo> edge_infos;
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    CHECK(dst_node->GetTaskType() == TaskType::kNormalMdUpdt);
    CompTaskNode* mdupdt_node = dynamic_cast<CompTaskNode*>(dst_node);
    for (TaskEdge* mdupdt_edge : mdupdt_node->out_edges()) {
      if (IsBackwardTaskType(mdupdt_edge->dst_node()->GetTaskType())) {
        CompTaskNode* bw_node = dynamic_cast<CompTaskNode*>(mdupdt_edge->dst_node());
        // There may be multiple out_regsts on the same edge for shared_model app
        EdgeInfo edge_info{bw_node->order_in_graph(), edge};
        edge_infos.emplace_back(edge_info);
      }
    }
  }
  std::sort(edge_infos.begin(), edge_infos.end(), [](const EdgeInfo& lhs, const EdgeInfo& rhs) {
    return lhs.bw_node_order < rhs.bw_node_order;
  });
  FOR_RANGE(size_t, idx, 0, edge_infos.size()) {
    std::string out_regst_name = "out_" + std::to_string(idx);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name, false, 1, 1);
    edge_infos[idx].edge->AddRegst(out_regst_name, out_regst);
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

  CompTaskNode* reduce_concat_node = FindPeerReduceConcatTaskNode();
  const auto& reduce_concat_consumed_regsts = reduce_concat_node->consumed_regsts();
  CHECK_EQ(reduce_concat_consumed_regsts.size(), produced_regsts().size());

  FOR_RANGE(size_t, i, 0, reduce_split_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    CHECK(out_regst.get() != nullptr);
    out_regst->CopyBlobDescFrom(
        reduce_concat_node->GetSoleConsumedRegst("in_" + std::to_string(i)).get());
    node->BindBnWithRegst(reduce_split_op->output_bns().Get(i), out_regst);
  }
}

CompTaskNode* ReduceSplitCompTaskNode::FindPeerReduceConcatTaskNode() {
  TaskNode* src_node = this;
  while (true) {
    for (TaskEdge* edge : src_node->in_edges()) {
      CompTaskNode* comp_task_node = dynamic_cast<CompTaskNode*>(edge->src_node());
      if (comp_task_node == nullptr) { continue; }
      if (comp_task_node->GetTaskType() == TaskType::kReduceConcat) {
        return comp_task_node;
      } else if (comp_task_node->GetTaskType() == TaskType::kNormalBackward) {
        LOG(FATAL) << "No peer ReduceConcat";
      } else {
        src_node = edge->src_node();
      }
    }
  }
  return nullptr;
}

}  // namespace oneflow
