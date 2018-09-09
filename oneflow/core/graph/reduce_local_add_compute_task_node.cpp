#include "oneflow/core/graph/reduce_local_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceLocalAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(out_edges().size(), machine_num);
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> succ_comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(succ_comp_task_nodes.size(), 1);
    int64_t out_parallel_id = succ_comp_task_nodes.front()->parallel_id();
    int64_t out_machine_rank = out_parallel_id / dev_num_of_each_machine;
    std::string regst_name = "out_" + std::to_string(out_machine_rank);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(regst_name, false, 1, 1);
    edge->AddRegst(regst_name, out_regst);
  }
}

void ReduceLocalAddCompTaskNode::ConsumeAllRegsts() {
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (src_node->GetTaskType() != TaskType::kReduceScatter) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    int64_t in_parallel_id = src_node->parallel_ctx()->parallel_id();
    int64_t in_device_rank = in_parallel_id % dev_num_of_each_machine;
    ConsumeRegst("in_" + std::to_string(in_device_rank), edge->GetSoleRegst());
  }
}

void ReduceLocalAddCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_local_add_conf;
  reduce_local_add_conf.set_name("reduce_local_add_" + NewUniqueId());
  reduce_local_add_conf.set_device_type(this->device_type());
  ReduceLocalAddOpConf* mut_local_add_conf = reduce_local_add_conf.mutable_reduce_local_add_conf();
  mut_local_add_conf->set_in_num(in_edges().size());
  mut_local_add_conf->set_out_num(out_edges().size());
  std::shared_ptr<Operator> reduce_local_add_op = ConstructOp(reduce_local_add_conf);
  node->mut_op() = reduce_local_add_op;

  FOR_RANGE(size_t, i, 0, reduce_local_add_op->input_bns().size()) {
    std::string in_name = reduce_local_add_op->input_bns().Get(i);
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst(in_name);
    node->BindBnWithRegst(in_name, in_regst);
  }
  FOR_RANGE(size_t, i, 0, reduce_local_add_op->output_bns().size()) {
    std::string out_name = reduce_local_add_op->output_bns().Get(i);
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(out_name);
    out_regst->AddLbi(reduce_local_add_op->BnInOp2Lbi(out_name));
    node->BindBnWithRegst(out_name, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void ReduceLocalAddCompTaskNode::EnableMemSharingInReduce(ReduceMemSharingCtx* ctx) {
  int64_t offset = ctx->Offset4ParallelId(parallel_id());
  int64_t scatter_count = produced_regsts().size();

  int64_t out_size = ctx->CtxIfScatter(scatter_count).ReduceSize();
  int64_t rank = ctx->Rank4ParallelId(parallel_id());
  int64_t offset_base = ctx->CtxIfGatherLast().Offset4ParallelId(parallel_id());

  FOR_RANGE(int, i, 0, scatter_count) {
    RegstDesc* out = GetProducedRegst("out_" + std::to_string(i)).get();
    ctx->EnableMemSharing4Regst(out, offset + i * out_size);
  }

  FOR_RANGE(int64_t, i, 0, ctx->LastCount()) {
    if (i == rank) { continue; }
    RegstDesc* consumed_regst = GetSoleConsumedRegst("in_" + std::to_string(i)).get();
    ctx->EnableMemSharing4Regst(consumed_regst, offset_base + i * ctx->ReduceSize());
  }

  std::vector<CompTaskNode*> scatter_on_in_edge;
  ForEachNodeOnInEdge([&](TaskNode* node) {
    if (node->GetTaskType() == kReduceScatter) {
      scatter_on_in_edge.push_back(dynamic_cast<CompTaskNode*>(node));
      return;
    }
  });
  CHECK_EQ(scatter_on_in_edge.size(), 1);
  BuildCtrlRegstBetweenReduceCopyNodes(scatter_on_in_edge.front(), this, ctx->LastCount() - 1);
}

}  // namespace oneflow
