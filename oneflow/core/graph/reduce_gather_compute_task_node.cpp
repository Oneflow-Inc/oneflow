#include "oneflow/core/graph/reduce_gather_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceGatherCompTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
}

void ReduceGatherCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (src_node->GetTaskType() != TaskType::kReduceGlobalAdd) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    CompTaskNode* reduce_global_add_node = dynamic_cast<CompTaskNode*>(src_node);
    ConsumeRegst("in_" + std::to_string(reduce_global_add_node->parallel_id()),
                 edge->GetSoleRegst());
  }
}

void ReduceGatherCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> reduce_gather_op = this->logical_node()->SoleOp();
  node->mut_op() = reduce_gather_op;
  FOR_RANGE(size_t, i, 0, reduce_gather_op->input_bns().size()) {
    node->BindBnWithRegst(reduce_gather_op->input_bns().Get(i),
                          GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(reduce_gather_op->BnInOp2Lbi(reduce_gather_op->SoleObn()));
  node->BindBnWithRegst(reduce_gather_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void ReduceGatherCompTaskNode::EnableMemSharingInReduce(
    std::function<void(RegstDesc* regst, int64_t offset)> EnableMemSharing4Regst) {
  EnableMemSharing4Regst(GetProducedRegst("out").get(), 0);
  for (const auto& kv : consumed_regsts()) {
    auto in_parallel_id = oneflow_cast<int64_t>(kv.first.substr(3));
    CHECK_EQ(1, kv.second.size());
    if (in_parallel_id == parallel_id()) { continue; }
    RegstDesc* regst = kv.second.front().get();
    EnableMemSharing4Regst(regst, InferRegstSize(*regst) * in_parallel_id);
  }

  std::vector<TaskNode*> global_add_on_in_edge;

  ForEachNodeOnInEdge([&](TaskNode* node) {
    if (node->GetTaskType() == kReduceGlobalAdd) { global_add_on_in_edge.push_back(node); }
  });

  CHECK_EQ(global_add_on_in_edge.size(), 1);

  TaskNode* global_add_copy_d2h = nullptr;
  for (TaskEdge* out_edge : global_add_on_in_edge.front()->out_edges()) {
    if (out_edge->dst_node()->GetTaskType() == TaskType::kCopyHd) {
      global_add_copy_d2h = out_edge->dst_node();
    }
  }

  for (TaskEdge* in_edge : this->in_edges()) {
    if (in_edge->src_node()->GetTaskType() == TaskType::kCopyHd) {
      global_add_copy_d2h->BuildCtrlRegstDesc(in_edge->src_node());
    }
  }
}

}  // namespace oneflow
