#include "oneflow/core/graph/reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceScatterCompTaskNode::ProduceAllRegstsAndBindEdges() {
  min_out_parallel_id_ = std::numeric_limits<int64_t>::max();
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(comp_task_nodes.size(), 1);
    int64_t parallel_id = comp_task_nodes.front()->parallel_id();
    min_out_parallel_id_ = std::min(min_out_parallel_id_, parallel_id);
    std::string out_regst_name = "out_" + std::to_string(parallel_id);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name, false);
    edge->AddRegst(out_regst_name, out_regst);
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
  reduce_scatter_op_conf.mutable_reduce_scatter_conf()->set_out_num(this->out_edges().size());
  std::shared_ptr<Operator> reduce_scatter_op = ConstructOp(reduce_scatter_op_conf);
  node->mut_op() = reduce_scatter_op;
  node->BindBnWithRegst(reduce_scatter_op->SoleIbn(), GetSoleConsumedRegst("in"));
  FOR_RANGE(size_t, i, 0, reduce_scatter_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst =
        GetProducedRegst("out_" + std::to_string(i + min_out_parallel_id_));
    const std::string& obn = reduce_scatter_op->output_bns().Get(i);
    out_regst->AddLbi(reduce_scatter_op->BnInOp2Lbi(obn));
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
