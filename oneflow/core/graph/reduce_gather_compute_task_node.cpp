#include "oneflow/core/graph/reduce_gather_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceGatherCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void ReduceGatherCompTaskNode::ConsumeAllRegsts() {
  std::vector<EdgeInfo> edge_infos;
  ForEachInDataEdge([&](TaskEdge* edge) {
    std::vector<CompTaskNode*> pred_comp_task_nodes = GetPredCompTaskNodesOnEdge(edge);
    CHECK_EQ(pred_comp_task_nodes.size(), 1);
    EdgeInfo edge_info = {edge, pred_comp_task_nodes.front()->task_id()};
    edge_infos.push_back(edge_info);
  });
  SortEdges(&edge_infos);
  FOR_RANGE(int64_t, in_edge_index, 0, edge_infos.size()) {
    ConsumeRegst("in_" + std::to_string(in_edge_index),
                 edge_infos[in_edge_index].edge->GetSoleRegst());
  }
}

void ReduceGatherCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_gather_op_conf;
  reduce_gather_op_conf.set_name(this->logical_node()->SoleOp()->op_name());
  reduce_gather_op_conf.set_device_type(this->device_type());
  reduce_gather_op_conf.mutable_reduce_gather_conf()->set_in_num(in_data_edges_size());
  std::shared_ptr<Operator> reduce_gather_op = ConstructOp(reduce_gather_op_conf, &GlobalJobDesc());
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

void ReduceGatherCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  const ReduceRankCtx& rank_ctx = GetRankCtx();
  int64_t base_offset = ctx.Offset4RankCtxParallelId(rank_ctx.CtxWithGather(), parallel_id());
  int64_t rank = rank_ctx.Rank4ParallelId(parallel_id());
  ctx.EnableMemSharing4Regst(GetProducedRegst("out").get(), base_offset);
  for (const auto& kv : consumed_regsts()) {
    auto in_parallel_id = oneflow_cast<int64_t>(kv.first.substr(3));
    CHECK_EQ(1, kv.second.size());
    if (in_parallel_id == rank) { continue; }
    RegstDesc* regst = kv.second.front().get();
    ctx.EnableMemSharing4Regst(regst,
                               base_offset + in_parallel_id * ctx.SegmentSize4RankCtx(rank_ctx));
  }

  TaskNode* nearest_add_task_node = FindPredReduceTaskNodeIf(
      [](TaskNode* node) { return node->GetTaskType() == TaskType::kReduceAdd; });
  CHECK(nearest_add_task_node);

  TaskNode* nearest_add_copy_d2h = nullptr;
  nearest_add_task_node->ForEachNodeOnOutEdge([&](TaskNode* node) {
    if (node->GetTaskType() == TaskType::kCopyHd) { nearest_add_copy_d2h = node; }
  });
  CHECK(nearest_add_copy_d2h);

  ForEachNodeOnInEdge([&](TaskNode* node) {
    if (node->GetTaskType() == TaskType::kCopyHd) {
      nearest_add_copy_d2h->BuildCtrlRegstDesc(node);
    }
  });
}

void ReduceGatherCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
