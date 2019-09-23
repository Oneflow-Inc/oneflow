#include "oneflow/core/graph/reduce_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void ReduceAddCompTaskNode::ConsumeAllRegsts() {
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

void ReduceAddCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_add_op_conf;
  reduce_add_op_conf.set_name(this->logical_node()->SoleOp()->op_name());
  reduce_add_op_conf.set_device_type(this->device_type());
  reduce_add_op_conf.mutable_reduce_add_conf()->set_in_num(in_data_edges_size());
  std::shared_ptr<Operator> reduce_add_op = ConstructOp(reduce_add_op_conf, &GlobalJobDesc());
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

void ReduceAddCompTaskNode::BuildCtrlRegstBetweenReduceCopyNodes(const CompTaskNode* src_reduce,
                                                                 const CompTaskNode* dst_reduce,
                                                                 int64_t copy_node_num) {
  struct ReduceCopyNodePair {
    TaskNode* copy_h2d;
    TaskNode* copy_d2h;
    ReduceCopyNodePair() : copy_h2d(nullptr), copy_d2h(nullptr) {}
  };
  HashMap<int64_t, ReduceCopyNodePair> mem_block_offset2copy_nodes;

  src_reduce->ForEachOutDataEdge([&](TaskEdge* out_edge) {
    if (out_edge->dst_node()->GetTaskType() == TaskType::kCopyHd) {
      int64_t offset = out_edge->GetSoleRegst()->mem_block_offset();
      mem_block_offset2copy_nodes[offset].copy_d2h = out_edge->dst_node();
    }
  });
  CHECK_EQ(copy_node_num, mem_block_offset2copy_nodes.size());

  dst_reduce->ForEachInDataEdge([&](TaskEdge* in_edge) {
    if (in_edge->src_node()->GetTaskType() == TaskType::kCopyHd) {
      int64_t offset = in_edge->GetSoleRegst()->mem_block_offset();
      CHECK(mem_block_offset2copy_nodes.find(offset) != mem_block_offset2copy_nodes.end());
      mem_block_offset2copy_nodes.at(offset).copy_h2d = in_edge->src_node();
    }
  });

  for (const auto& kv : mem_block_offset2copy_nodes) {
    kv.second.copy_d2h->BuildCtrlRegstDesc(kv.second.copy_h2d);
  }
}

void ReduceAddCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
