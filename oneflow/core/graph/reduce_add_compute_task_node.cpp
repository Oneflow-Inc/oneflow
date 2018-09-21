#include "oneflow/core/graph/reduce_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 1, 1);
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedRegst(edge, "out"); }
}

void ReduceAddCompTaskNode::ConsumeAllRegsts() {
  std::vector<EdgeInfo> edge_infos;
  for (TaskEdge* edge : in_edges()) {
    std::vector<CompTaskNode*> pred_comp_task_nodes = GetPredCompTaskNodesOnEdge(edge);
    CHECK_EQ(pred_comp_task_nodes.size(), 1);
    EdgeInfo edge_info = {edge, pred_comp_task_nodes.front()->task_id()};
    edge_infos.push_back(edge_info);
  }
  SortEdges(&edge_infos);
  FOR_RANGE(int64_t, in_edge_index, 0, edge_infos.size()) {
    ConsumeRegst("in_" + std::to_string(in_edge_index),
                 edge_infos[in_edge_index].edge->GetSoleRegst());
  }
}

void ReduceAddCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_add_op_conf;
  reduce_add_op_conf.set_name("reduce_add_" + NewUniqueId());
  reduce_add_op_conf.set_device_type(this->device_type());
  reduce_add_op_conf.mutable_reduce_add_conf()->set_in_num(in_edges().size());
  std::shared_ptr<Operator> reduce_add_op = ConstructOp(reduce_add_op_conf);
  node->mut_op() = reduce_add_op;
  for (const std::string& input_bn : reduce_add_op->input_bns()) {
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst(input_bn);
    node->BindBnWithRegst(input_bn, in_regst);
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(reduce_add_op->BnInOp2Lbi(reduce_add_op->SoleObn()));
  node->BindBnWithRegst(reduce_add_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void ReduceAddCompTaskNode::EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) {
  const ReduceRankCtx& rank_ctx = GetRankCtx();
  int64_t offset = ctx.Offset4RankCtxParallelId(rank_ctx.CtxWithGather(), parallel_id());
  int64_t rank = rank_ctx.Rank4ParallelId(parallel_id());
  int64_t segment_size = ctx.SegmentSize4RankCtx(rank_ctx);

  ctx.EnableMemSharing4Regst(GetProducedRegst("out").get(),
                             ctx.Offset4RankCtxParallelId(rank_ctx, parallel_id()));
  for (const auto& kv : consumed_regsts()) {
    auto in_parallel_id = oneflow_cast<int64_t>(kv.first.substr(3));
    CHECK_EQ(1, kv.second.size());
    if (in_parallel_id == rank) { continue; }
    RegstDesc* regst = kv.second.front().get();
    ctx.EnableMemSharing4Regst(regst, offset + in_parallel_id * segment_size);
  }

  std::vector<CompTaskNode*> scatter_on_in_edge;
  ForEachNodeOnInEdge([&](TaskNode* node) {
    if (node->GetTaskType() == kReduceScatter) {
      scatter_on_in_edge.push_back(dynamic_cast<CompTaskNode*>(node));
      return;
    }
  });
  CHECK_EQ(scatter_on_in_edge.size(), 1);
  if (!scatter_on_in_edge.empty()) {
    BuildCtrlRegstBetweenReduceCopyNodes(scatter_on_in_edge.front(), this,
                                         rank_ctx.StageSegmentCount() - 1);
  }
}

}  // namespace oneflow
