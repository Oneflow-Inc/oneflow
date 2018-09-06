#include "oneflow/core/graph/reduce_local_add2_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceLocalAdd2CompTaskNode::ProduceAllRegstsAndBindEdges() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(out_edges().size(), machine_num);
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> succ_comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(succ_comp_task_nodes.size(), 1);
    int64_t parallel_id = succ_comp_task_nodes.front()->parallel_id();
    int64_t out_edge_index = parallel_id / dev_num_of_each_machine;
    std::string regst_name = "out_" + std::to_string(out_edge_index);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(regst_name, false, 1, 1);
    edge->AddRegst(regst_name, out_regst);
  }
}

void ReduceLocalAdd2CompTaskNode::ConsumeAllRegsts() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());

  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (src_node->GetTaskType() != TaskType::kReduceScatter2) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    int64_t parallel_id = src_node->parallel_ctx()->parallel_id();
    int64_t in_edge_index = parallel_id % dev_num_of_each_machine;
    ConsumeRegst("in_" + std::to_string(in_edge_index), edge->GetSoleRegst());
  }
}

void ReduceLocalAdd2CompTaskNode::BuildExecGphAndRegst() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());

  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_local_add2_conf;
  reduce_local_add2_conf.set_name("reduce_local_add2_" + NewUniqueId());
  reduce_local_add2_conf.set_device_type(this->device_type());
  ReduceLocalAdd2OpConf* mut_local_add_conf =
      reduce_local_add2_conf.mutable_reduce_local_add2_conf();
  mut_local_add_conf->set_in_num(in_edges().size());
  mut_local_add_conf->set_out_num(out_edges().size());
  std::shared_ptr<Operator> reduce_local_add2_op = ConstructOp(reduce_local_add2_conf);
  node->mut_op() = reduce_local_add2_op;

  FOR_RANGE(size_t, i, 0, reduce_local_add2_op->input_bns().size()) {
    std::string in_name = reduce_local_add2_op->input_bns().Get(i);
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst(in_name);
    node->BindBnWithRegst(in_name, in_regst);
  }
  FOR_RANGE(size_t, i, 0, reduce_local_add2_op->output_bns().size()) {
    std::string out_name = reduce_local_add2_op->output_bns().Get(i);
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(out_name);
    out_regst->AddLbi(reduce_local_add2_op->BnInOp2Lbi(out_name));
    node->BindBnWithRegst(out_name, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
