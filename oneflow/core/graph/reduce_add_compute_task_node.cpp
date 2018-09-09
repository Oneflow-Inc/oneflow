#include "oneflow/core/graph/reduce_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 1, 1);
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedRegst(edge, "out"); }
}

void ReduceAddCompTaskNode::ConsumeAllRegsts() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());
  bool has_local_reduce = machine_num > 1 && dev_num_of_each_machine > 1;

  for (TaskEdge* edge : in_edges()) {
    std::vector<CompTaskNode*> pred_comp_task_nodes = GetPredCompTaskNodesOnEdge(edge);
    CHECK_EQ(pred_comp_task_nodes.size(), 1);
    int64_t in_parallel_id = pred_comp_task_nodes.front()->parallel_id();
    int64_t in_machine_rank = in_parallel_id / dev_num_of_each_machine;
    int64_t in_edge_index = has_local_reduce ? in_machine_rank : in_parallel_id;
    ConsumeRegst("in_" + std::to_string(in_edge_index), edge->GetSoleRegst());
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

void ReduceAddCompTaskNode::EnableMemSharingInReduce(ReduceMemSharingCtx* ctx) {
  ReduceMemSharingCtx ctx_if_gather = ctx->CtxIfGatherLast();
  int64_t offset = ctx_if_gather.Offset4ParallelId(parallel_id());
  int64_t rank = ctx->Rank4ParallelId(parallel_id());
  int64_t reduce_size = ctx->ReduceSize();

  ctx->EnableMemSharing4Regst(GetProducedRegst("out").get(), ctx->Offset4ParallelId(parallel_id()));
  for (const auto& kv : consumed_regsts()) {
    auto in_parallel_id = oneflow_cast<int64_t>(kv.first.substr(3));
    CHECK_EQ(1, kv.second.size());
    if (in_parallel_id == rank) { continue; }
    RegstDesc* regst = kv.second.front().get();
    ctx->EnableMemSharing4Regst(regst, offset + in_parallel_id * reduce_size);
  }

  std::vector<CompTaskNode*> local_add_on_in_edge;
  std::vector<CompTaskNode*> scatter_on_in_edge;
  ForEachNodeOnInEdge([&](TaskNode* node) {
    if (node->GetTaskType() == kReduceLocalAdd) {
      local_add_on_in_edge.push_back(dynamic_cast<CompTaskNode*>(node));
      return;
    }
    if (node->GetTaskType() == kReduceScatter) {
      scatter_on_in_edge.push_back(dynamic_cast<CompTaskNode*>(node));
      return;
    }
  });
  CHECK_EQ(local_add_on_in_edge.size() + scatter_on_in_edge.size(), 1);
  if (!local_add_on_in_edge.empty()) {
    BuildCtrlRegstBetweenReduceCopyNodes(local_add_on_in_edge.front(), this, ctx->LastCount() - 1);
  }
  if (!scatter_on_in_edge.empty()) {
    BuildCtrlRegstBetweenReduceCopyNodes(scatter_on_in_edge.front(), this, ctx->LastCount() - 1);
  }
}

}  // namespace oneflow
