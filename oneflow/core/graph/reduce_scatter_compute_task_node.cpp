#include "oneflow/core/graph/reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceScatterCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::vector<EdgeInfo> edge_infos;
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(comp_task_nodes.size(), 1);
    EdgeInfo edge_info = {edge, comp_task_nodes.front()->task_id()};
    edge_infos.push_back(edge_info);
  }
  std::sort(edge_infos.begin(), edge_infos.end(),
            [](const EdgeInfo& lhs, const EdgeInfo& rhs) { return lhs.order < rhs.order; });
  FOR_RANGE(int64_t, out_edge_index, 0, edge_infos.size()) {
    std::string out_regst_name = "out_" + std::to_string(out_edge_index);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name, false, 1, 1);
    edge_infos[out_edge_index].edge->AddRegst(out_regst_name, out_regst);
  }
}

void ReduceScatterCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", this->SoleInEdge()->GetSoleRegst());
}

void ReduceScatterCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_scatter_op_conf;
  reduce_scatter_op_conf.set_name("reduce_scatter_" + NewUniqueId());
  reduce_scatter_op_conf.set_device_type(this->device_type());
  reduce_scatter_op_conf.mutable_reduce_scatter_conf()->set_out_num(out_edges().size());

  std::shared_ptr<Operator> reduce_scatter_op = ConstructOp(reduce_scatter_op_conf);
  node->mut_op() = reduce_scatter_op;
  node->BindBnWithRegst(reduce_scatter_op->SoleIbn(), GetSoleConsumedRegst("in"));

  FOR_RANGE(size_t, i, 0, reduce_scatter_op->output_bns().size()) {
    std::string out_name = "out_" + std::to_string(i);
    CHECK_EQ(out_name, reduce_scatter_op->output_bns().Get(i));
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(out_name);
    CHECK(out_regst.get() != nullptr);
    out_regst->AddLbi(reduce_scatter_op->BnInOp2Lbi(out_name));
    node->BindBnWithRegst(out_name, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void ReduceScatterCompTaskNode::EnableMemSharingInReduce(ReduceMemSharingCtx* ctx) {
  int64_t offset = ctx->Offset4ParallelId(parallel_id());
  int64_t reduce_size = ctx->ReduceSize();
  int64_t scatter_count = produced_regsts().size();
  CHECK_EQ(reduce_size % scatter_count, 0);
  int64_t out_size = reduce_size / scatter_count;
  FOR_RANGE(int64_t, i, 0, scatter_count) {
    RegstDesc* out = GetProducedRegst("out_" + std::to_string(i)).get();
    ctx->EnableMemSharing4Regst(out, offset + i * out_size);
  }
  ctx->Scatter(scatter_count);
  if (this->SoleInEdge()->src_node()->GetTaskType() == TaskType::kReduceConcat) { return; }
  ctx->EnableMemSharing4Regst(GetSoleConsumedRegst("in").get(), offset);
}

}  // namespace oneflow
