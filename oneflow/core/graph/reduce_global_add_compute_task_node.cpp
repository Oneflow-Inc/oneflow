#include "oneflow/core/graph/reduce_global_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceGlobalAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 1, 1);
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedRegst(edge, "out"); }
}

void ReduceGlobalAddCompTaskNode::ConsumeAllRegsts() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());
  bool do_local_reduce_scatter = machine_num > 1 && dev_num_of_each_machine > 1;

  for (TaskEdge* edge : in_edges()) {
    std::vector<CompTaskNode*> pred_comp_task_nodes = GetPredCompTaskNodesOnEdge(edge);
    CHECK_EQ(pred_comp_task_nodes.size(), 1);
    int64_t in_parallel_id = pred_comp_task_nodes.front()->parallel_id();
    int64_t in_machine_rank = in_parallel_id / dev_num_of_each_machine;
    int64_t in_edge_index = do_local_reduce_scatter ? in_machine_rank : in_parallel_id;
    ConsumeRegst("in_" + std::to_string(in_edge_index), edge->GetSoleRegst());
  }
}

void ReduceGlobalAddCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_global_add_op_conf;
  reduce_global_add_op_conf.set_name("reduce_global_add_" + NewUniqueId());
  reduce_global_add_op_conf.set_device_type(this->device_type());
  reduce_global_add_op_conf.mutable_reduce_global_add_conf()->set_in_num(in_edges().size());
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

}  // namespace oneflow
