#include "oneflow/core/graph/reduce_global_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceGlobalAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 1, 1);
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedRegst(edge, "out"); }
}

void ReduceGlobalAddCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    std::vector<CompTaskNode*> pred_comp_task_nodes = GetPredCompTaskNodesOnEdge(edge);
    CHECK_EQ(pred_comp_task_nodes.size(), 1);
    int64_t parallel_id = pred_comp_task_nodes.front()->parallel_id();
    ConsumeRegst("in_" + std::to_string(parallel_id), edge->GetSoleRegst());
    in_parallel_ids_.Add(parallel_id);
  }
}

void ReduceGlobalAddCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_global_add_op_conf;
  reduce_global_add_op_conf.set_name("reduce_global_add_" + NewUniqueId());
  reduce_global_add_op_conf.set_device_type(this->device_type());
  *reduce_global_add_op_conf.mutable_reduce_global_add_conf()->mutable_in_parallel_ids() =
      in_parallel_ids_;
  std::shared_ptr<Operator> reduce_global_add_op = ConstructOp(reduce_global_add_op_conf);
  node->mut_op() = reduce_global_add_op;
  for (const std::string& input_bn : reduce_global_add_op->input_bns()) {
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst(input_bn);
    node->BindBnWithRegst(input_bn, in_regst);
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(reduce_global_add_op->BnInOp2Lbi(reduce_global_add_op->SoleObn()));
  node->BindBnWithRegst(reduce_global_add_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void ReduceGlobalAddCompTaskNode::EnableMemSharingInReduce(
    std::function<void(RegstDesc* regst, int64_t offset)> EnableMemSharing4Regst) {
  EnableMemSharing4Regst(GetProducedRegst("out").get(),
                         parallel_id() * InferRegstSize(*GetProducedRegst("out")));
  for (const auto& kv : consumed_regsts()) {
    auto in_parallel_id = oneflow_cast<int64_t>(kv.first.substr(3));
    CHECK_EQ(1, kv.second.size());
    if (in_parallel_id == parallel_id()) { continue; }
    RegstDesc* regst = kv.second.front().get();
    EnableMemSharing4Regst(regst, InferRegstSize(*regst) * in_parallel_id);
  }
}

}  // namespace oneflow
