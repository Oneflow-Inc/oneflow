#include "oneflow/core/graph/reduce_local_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceLocalAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> succ_comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(succ_comp_task_nodes.size(), 1);
    std::string regst_name = "out_" + std::to_string(succ_comp_task_nodes.front()->machine_id());
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(regst_name, false);
    edge->AddRegst(regst_name, out_regst);
  }
}

void ReduceLocalAddCompTaskNode::ConsumeAllRegsts() {
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    TaskEdge* in_edge = edge;
    while (in_edge->src_node()->GetTaskType() != TaskType::kReduceScatter) {
      in_edge = in_edge->src_node()->SoleInEdge();
    }
    const auto& name_in_producer2regst = in_edge->name_in_producer2regst();
    CHECK_EQ(1, name_in_producer2regst.size());
    const std::string& name_in_scatter = name_in_producer2regst.begin()->first;

    int64_t index_of_the_out_regst = oneflow_cast<int64_t>(name_in_scatter.substr(4));
    int64_t index_of_the_out_regst_from_this_scatter =
        index_of_the_out_regst / dev_num_of_each_machine;

    int64_t parallel_id = in_edge->src_node()->parallel_ctx()->parallel_id();
    int64_t src_dev_index_of_this_machine = parallel_id % dev_num_of_each_machine;

    int64_t in_regst_index = index_of_the_out_regst_from_this_scatter * dev_num_of_each_machine
                             + src_dev_index_of_this_machine;
    ConsumeRegst("in_" + std::to_string(in_regst_index), edge->GetSoleRegst());
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
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in_" + std::to_string(i));
    node->BindBnWithRegst(reduce_local_add_op->input_bns().Get(i), in_regst);
  }
  FOR_RANGE(size_t, i, 0, reduce_local_add_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    out_regst->AddLbi(reduce_local_add_op->BnInOp2Lbi(reduce_local_add_op->output_bns().Get(i)));
    node->BindBnWithRegst(reduce_local_add_op->output_bns().Get(i), out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
