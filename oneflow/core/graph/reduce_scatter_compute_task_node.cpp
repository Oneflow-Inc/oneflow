#include "oneflow/core/graph/reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceScatterCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());

  std::vector<int64_t> edge_index4dst_dev(dev_num_of_each_machine, 0);
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(comp_task_nodes.size(), 1);
    int64_t parallel_id = comp_task_nodes.front()->parallel_ctx()->parallel_id();

    int64_t out_edge_index = -1;
    if (machine_num == parallel_ctx()->parallel_num()) {
      out_edge_index = parallel_id;
    } else {
      int64_t dst_dev_index_of_this_machine = parallel_id % dev_num_of_each_machine;
      int64_t edge_index_of_this_dst_dev = edge_index4dst_dev.at(dst_dev_index_of_this_machine);
      edge_index4dst_dev.at(dst_dev_index_of_this_machine) += 1;
      CHECK_LE(edge_index4dst_dev.at(dst_dev_index_of_this_machine), machine_num);

      out_edge_index =
          edge_index_of_this_dst_dev * dev_num_of_each_machine + dst_dev_index_of_this_machine;
    }
    std::string out_regst_name = "out_" + std::to_string(out_edge_index);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(out_regst_name, false, 1, 1);
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
  reduce_scatter_op_conf.mutable_reduce_scatter_conf()->set_out_num(out_edges().size());

  std::shared_ptr<Operator> reduce_scatter_op = ConstructOp(reduce_scatter_op_conf);
  node->mut_op() = reduce_scatter_op;
  node->BindBnWithRegst(reduce_scatter_op->SoleIbn(), GetSoleConsumedRegst("in"));

  FOR_RANGE(size_t, i, 0, reduce_scatter_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    CHECK(out_regst.get() != nullptr);
    const std::string& obn = reduce_scatter_op->output_bns().Get(i);
    out_regst->AddLbi(reduce_scatter_op->BnInOp2Lbi(obn));
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void ReduceScatterCompTaskNode::EnableMemSharingInReduce(
    std::function<void(RegstDesc* regst, int64_t offset)> EnableMemSharing4Regst) {
  FOR_RANGE(int64_t, i, 0, parallel_ctx()->parallel_num()) {
    RegstDesc* out = GetProducedRegst("out_" + std::to_string(i)).get();
    EnableMemSharing4Regst(out, i * InferRegstSize(*out));
  }

  if (this->SoleInEdge()->src_node()->GetTaskType() == TaskType::kReduceConcat) { return; }

  EnableMemSharing4Regst(GetSoleConsumedRegst("in").get(), 0);
}

}  // namespace oneflow
