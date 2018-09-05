#include "oneflow/core/graph/reduce_add2_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceAdd2CompTaskNode::ProduceAllRegstsAndBindEdges() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());
  bool do_local_reduce_scatter = machine_num > 1 && dev_num_of_each_machine > 1;
  if (do_local_reduce_scatter) {
    CHECK_EQ(out_edges().size(), machine_num);
  } else {
    CHECK_EQ(out_edges().size(), parallel_ctx()->parallel_num());
  }
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> succ_comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(succ_comp_task_nodes.size(), 1);
    int64_t parallel_id = succ_comp_task_nodes.front()->parallel_id();
    int64_t out_edge_index =
        do_local_reduce_scatter ? parallel_id / dev_num_of_each_machine : parallel_id;
    std::string regst_name = "out_" + std::to_string(out_edge_index);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(regst_name, false, 1, 1);
    edge->AddRegst(regst_name, out_regst);
  }
}

void ReduceAdd2CompTaskNode::ConsumeAllRegsts() {
  // TODO(jiyuan): use regst name: in_0, in_1, ...
  for (TaskEdge* edge : in_edges()) { ConsumeRegst("in", edge->GetSoleRegst()); }
}

void ReduceAdd2CompTaskNode::BuildExecGphAndRegst() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());

  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_add2_conf;
  reduce_add2_conf.set_name("reduce_add2_" + NewUniqueId());
  reduce_add2_conf.set_device_type(this->device_type());
  ReduceAdd2OpConf* mut_local_add_conf = reduce_add2_conf.mutable_reduce_add2_conf();
  mut_local_add_conf->set_in_num(in_edges().size());
  mut_local_add_conf->set_out_num(out_edges().size());
  std::shared_ptr<Operator> reduce_add2_op = ConstructOp(reduce_add2_conf);
  node->mut_op() = reduce_add2_op;

  BindIbnWithInRegst();
  FOR_RANGE(size_t, i, 0, reduce_add2_op->output_bns().size()) {
    std::string out_name = reduce_add2_op->output_bns().Get(i);
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst(out_name);
    out_regst->AddLbi(reduce_add2_op->BnInOp2Lbi(out_name));
    node->BindBnWithRegst(out_name, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void ReduceAdd2CompTaskNode::BindIbnWithInRegst() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());
  bool do_local_reduce_scatter = machine_num > 1 && dev_num_of_each_machine > 1;

  ExecNode* node = mut_exec_gph().SoleNode();
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (dynamic_cast<CompTaskNode*>(src_node) == nullptr) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    bool is_local_add = src_node->GetTaskType() == TaskType::kReduceScatter2;
    int64_t parallel_id = src_node->parallel_ctx()->parallel_id();
    int64_t in_edge_index = do_local_reduce_scatter
                                ? (is_local_add ? parallel_id % dev_num_of_each_machine
                                                : parallel_id / dev_num_of_each_machine)
                                : parallel_id;

    std::shared_ptr<RegstDesc> in_regst = edge->GetSoleRegst();
    CHECK_EQ(1, in_regst->NumOfLbi());
    node->BindBnWithRegst(node->op()->input_bns().Get(in_edge_index), in_regst);
  }
}

}  // namespace oneflow
