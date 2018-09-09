#include "oneflow/core/graph/reduce_gather_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceGatherCompTaskNode::ProduceAllRegstsAndBindEdges() {
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 1, 1);
  for (TaskEdge* edge : out_edges()) { edge->AddRegst("out", out_regst); }
}

void ReduceGatherCompTaskNode::ConsumeAllRegsts() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());
  bool has_local_reduce = machine_num > 1 && dev_num_of_each_machine > 1;

  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (dynamic_cast<CompTaskNode*>(src_node) == nullptr) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    bool is_local_gather = src_node->GetTaskType() == TaskType::kReduceGather;
    int64_t in_parallel_id = src_node->parallel_ctx()->parallel_id();
    int64_t in_device_rank = in_parallel_id % dev_num_of_each_machine;
    int64_t in_machine_rank = in_parallel_id / dev_num_of_each_machine;
    int64_t in_edge_index =
        has_local_reduce ? (is_local_gather ? in_device_rank : in_machine_rank) : in_parallel_id;
    ConsumeRegst("in_" + std::to_string(in_edge_index), edge->GetSoleRegst());
  }
}

void ReduceGatherCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_gather_op_conf;
  reduce_gather_op_conf.set_name("reduce_gather_" + NewUniqueId());
  reduce_gather_op_conf.set_device_type(this->device_type());
  reduce_gather_op_conf.mutable_reduce_gather_conf()->set_in_num(in_edges().size());
  std::shared_ptr<Operator> reduce_gather_op = ConstructOp(reduce_gather_op_conf);
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

void ReduceGatherCompTaskNode::EnableMemSharingInReduce(ReduceMemSharingCtx* ctx) {
  int64_t base_offset = ctx->CtxIfGatherLast().Offset4ParallelId(parallel_id());
  int64_t rank = ctx->Rank4ParallelId(parallel_id());
  ctx->EnableMemSharing4Regst(GetProducedRegst("out").get(), base_offset);
  for (const auto& kv : consumed_regsts()) {
    auto in_parallel_id = oneflow_cast<int64_t>(kv.first.substr(3));
    CHECK_EQ(1, kv.second.size());
    if (in_parallel_id == rank) { continue; }
    RegstDesc* regst = kv.second.front().get();
    ctx->EnableMemSharing4Regst(regst, base_offset + in_parallel_id * ctx->ReduceSize());
  }

  ctx->Gather(ctx->LastCount());
  std::vector<TaskNode*> global_add_on_in_edge;
  ForEachNodeOnInEdge([&](TaskNode* node) {
    if (node->GetTaskType() == kReduceGlobalAdd) { global_add_on_in_edge.push_back(node); }
  });

  // If not local gather
  if (global_add_on_in_edge.size()) {
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
}

}  // namespace oneflow
