#include "oneflow/core/graph/reduce_local_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceLocalAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> succ_comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(succ_comp_task_nodes.size(), 1);
    std::string regst_name = "out_" + std::to_string(succ_comp_task_nodes.front()->machine_id());
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(regst_name, false, 1, 1);
    edge->AddRegst(regst_name, out_regst);
  }
}

void ReduceLocalAddCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) { ConsumeRegst("in", edge->GetSoleRegst()); }
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

  BindIbnWithInRegst();
  FOR_RANGE(size_t, i, 0, reduce_local_add_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    out_regst->AddLbi(reduce_local_add_op->BnInOp2Lbi(reduce_local_add_op->output_bns().Get(i)));
    node->BindBnWithRegst(reduce_local_add_op->output_bns().Get(i), out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void ReduceLocalAddCompTaskNode::BindIbnWithInRegst() {
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  ExecNode* node = mut_exec_gph().SoleNode();
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (src_node->GetTaskType() != TaskType::kReduceScatter) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    int64_t parallel_id = src_node->parallel_ctx()->parallel_id();
    int64_t src_dev_index_of_this_machine = parallel_id % dev_num_of_each_machine;

    std::shared_ptr<RegstDesc> in_regst = edge->GetSoleRegst();
    CHECK_EQ(1, in_regst->NumOfLbi());
    int64_t index_of_lbi_in_scatter = -1;
    in_regst->ForEachLbi([&index_of_lbi_in_scatter](const LogicalBlobId& lbi) {
      index_of_lbi_in_scatter = oneflow_cast<int64_t>(lbi.blob_name().substr(4));
    });
    int64_t index_of_lbi_from_the_dev = index_of_lbi_in_scatter / dev_num_of_each_machine;

    int64_t ibn_index =
        index_of_lbi_from_the_dev * dev_num_of_each_machine + src_dev_index_of_this_machine;
    node->BindBnWithRegst(node->op()->input_bns().Get(ibn_index), in_regst);
  }
}

void ReduceLocalAddCompTaskNode::EnableMemSharingInReduce(
    std::function<void(RegstDesc* regst, int64_t offset)> EnableMemSharing4Regst) {
  FOR_RANGE(int, i, 0, logical_node()->parallel_desc()->sorted_machine_ids().size()) {
    RegstDesc* out = GetProducedRegst("out_" + std::to_string(i)).get();
    EnableMemSharing4Regst(out,
                           (i * logical_node()->parallel_desc()->device_num_of_each_machine()
                            + logical_node()->parallel_desc()->DeviceRank4ParallelId(parallel_id()))
                               * InferRegstSize(*out));
  }

  ExecNode* local_add_exec_node = exec_gph().SoleNode();

  FOR_RANGE(int64_t, i, 0, parallel_ctx()->parallel_num()) {
    if (logical_node()->parallel_desc()->DeviceRank4ParallelId(i)
        == logical_node()->parallel_desc()->DeviceRank4ParallelId(parallel_id())) {
      continue;
    }

    RegstDesc* consumed_regst =
        local_add_exec_node->RegstDesc4BnInOp(local_add_exec_node->op()->input_bns().Get(i));
    EnableMemSharing4Regst(consumed_regst, InferRegstSize(*consumed_regst) * i);
  }
}

}  // namespace oneflow
