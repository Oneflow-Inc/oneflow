#include "oneflow/core/graph/reduce_local_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceLocalAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  min_out_parallel_id_ = std::numeric_limits<int64_t>::max();
  for (TaskEdge* edge : out_edges()) {
    std::vector<CompTaskNode*> succ_comp_task_nodes = GetSuccCompTaskNodesOnEdge(edge);
    CHECK_EQ(succ_comp_task_nodes.size(), 1);
    int64_t parallel_id = succ_comp_task_nodes.front()->parallel_id();
    min_out_parallel_id_ = std::min(min_out_parallel_id_, parallel_id);
    std::string regst_name = "out_" + std::to_string(parallel_id);
    std::shared_ptr<RegstDesc> out_regst = ProduceRegst(regst_name);
    edge->AddRegst(regst_name, out_regst);
  }
  ProduceRegst("data_tmp", 1, 1);
}

void ReduceLocalAddCompTaskNode::ConsumeAllRegsts() {
  min_in_parallel_id_ = std::numeric_limits<int64_t>::max();
  for (TaskEdge* edge : in_edges()) {
    std::vector<CompTaskNode*> pred_comp_task_nodes = GetPredCompTaskNodesOnEdge(edge);
    CHECK_EQ(pred_comp_task_nodes.size(), 1);
    int64_t parallel_id = pred_comp_task_nodes.front()->parallel_id();
    min_in_parallel_id_ = std::min(min_in_parallel_id_, parallel_id);
    ConsumeRegst("in_" + std::to_string(parallel_id), edge->GetSoleRegst());
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
  mut_local_add_conf->set_min_in_parallel_id(min_in_parallel_id_);
  mut_local_add_conf->set_min_out_parallel_id(min_out_parallel_id_);
  TaskNode* pred_task_node = (*in_edges().begin())->src_node();
  while (pred_task_node->GetTaskType() != TaskType::kReduceScatter) {
    pred_task_node = pred_task_node->SoleInEdge()->src_node();
  }
  std::shared_ptr<RegstDesc> diff_acc_regst =
      pred_task_node->consumed_regsts().begin()->second.front().lock();
  mut_local_add_conf->set_model_elem_cnt(
      diff_acc_regst->GetBlobDesc(GenPackedLbi())->shape().elem_cnt());
  std::shared_ptr<Operator> reduce_local_add_op = ConstructOp(reduce_local_add_conf);
  node->mut_op() = reduce_local_add_op;
  FOR_RANGE(size_t, i, 0, reduce_local_add_op->input_bns().size()) {
    std::shared_ptr<RegstDesc> in_regst =
        GetSoleConsumedRegst("in_" + std::to_string(i + min_in_parallel_id_));
    node->BindBnWithRegst(reduce_local_add_op->input_bns().Get(i), in_regst);
  }
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, GetProducedRegst("data_tmp"));
  FOR_RANGE(size_t, i, 0, reduce_local_add_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst =
        GetProducedRegst("out_" + std::to_string(i + min_out_parallel_id_));
    out_regst->AddLbi(reduce_local_add_op->BnInOp2Lbi(reduce_local_add_op->output_bns().Get(i)));
    node->BindBnWithRegst(reduce_local_add_op->output_bns().Get(i), out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
