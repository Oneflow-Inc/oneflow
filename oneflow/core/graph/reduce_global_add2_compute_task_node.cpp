#include "oneflow/core/graph/reduce_global_add2_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceGlobalAdd2CompTaskNode::ProduceAllRegstsAndBindEdges() {
  /*
  ProduceRegst("out", false, 1, 1);
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedRegst(edge, "out"); }
  */
}

void ReduceGlobalAdd2CompTaskNode::ConsumeAllRegsts() {
  /*
  for (TaskEdge* edge : in_edges()) {
    std::vector<CompTaskNode*> pred_comp_task_nodes = GetPredCompTaskNodesOnEdge(edge);
    CHECK_EQ(pred_comp_task_nodes.size(), 1);
    int64_t parallel_id = pred_comp_task_nodes.front()->parallel_id();
    ConsumeRegst("in_" + std::to_string(parallel_id), edge->GetSoleRegst());
    in_parallel_ids_.Add(parallel_id);
  }
  */
}

void ReduceGlobalAdd2CompTaskNode::BuildExecGphAndRegst() {
  /*
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_global_add2_op_conf;
  reduce_global_add2_op_conf.set_name("reduce_global_add2_" + NewUniqueId());
  reduce_global_add2_op_conf.set_device_type(this->device_type());
  *reduce_global_add2_op_conf.mutable_reduce_global_add2_conf()->mutable_in_parallel_ids() =
      in_parallel_ids_;
  std::shared_ptr<Operator> reduce_global_add2_op = ConstructOp(reduce_global_add2_op_conf);
  node->mut_op() = reduce_global_add2_op;
  for (const std::string& input_bn : reduce_global_add2_op->input_bns()) {
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst(input_bn);
    node->BindBnWithRegst(input_bn, in_regst);
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(reduce_global_add2_op->BnInOp2Lbi(reduce_global_add2_op->SoleObn()));
  node->BindBnWithRegst(reduce_global_add2_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
  */
}

}  // namespace oneflow
