#include "oneflow/core/graph/reduce_scatter2_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceScatter2CompTaskNode::ProduceAllRegstsAndBindEdges() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());
  bool do_local_reduce_scatter = machine_num > 1 && dev_num_of_each_machine > 1;

  if (do_local_reduce_scatter) {
    CHECK_EQ(out_edges().size(), dev_num_of_each_machine);
  } else {
    CHECK_EQ(out_edges().size(), parallel_ctx()->parallel_num());
  }
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(comp_task_nodes.size(), 1);
    int64_t parallel_id = comp_task_nodes.front()->parallel_ctx()->parallel_id();
    int64_t out_edge_index =
        do_local_reduce_scatter ? parallel_id % dev_num_of_each_machine : parallel_id;
    std::string out_regst_name = "out_" + std::to_string(out_edge_index);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name, false, 1, 1);
    edge->AddRegst(out_regst_name, out_regst);
  }
}

void ReduceScatter2CompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", this->SoleInEdge()->GetSoleRegst());
}

void ReduceScatter2CompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_scatter2_op_conf;
  reduce_scatter2_op_conf.set_name("reduce_scatter2_" + NewUniqueId());
  reduce_scatter2_op_conf.set_device_type(this->device_type());
  reduce_scatter2_op_conf.mutable_reduce_scatter2_conf()->set_out_num(out_edges().size());

  std::shared_ptr<Operator> reduce_scatter2_op = ConstructOp(reduce_scatter2_op_conf);
  node->mut_op() = reduce_scatter2_op;
  node->BindBnWithRegst(reduce_scatter2_op->SoleIbn(), GetSoleConsumedRegst("in"));

  FOR_RANGE(size_t, i, 0, reduce_scatter2_op->output_bns().size()) {
    std::string out_name = "out_" + std::to_string(i);
    CHECK_EQ(out_name, reduce_scatter2_op->output_bns().Get(i));
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(out_name);
    CHECK(out_regst.get() != nullptr);
    out_regst->AddLbi(reduce_scatter2_op->BnInOp2Lbi(out_name));
    node->BindBnWithRegst(out_name, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
