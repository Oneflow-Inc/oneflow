#include "oneflow/core/graph/reduce_local_add_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceLocalAddCompTaskNode::ProduceAllRegstsAndBindEdges() {
  min_out_parallel_id_ = std::numeric_limits<int64_t>::max();
  HashMap<TaskEdge*, int64_t> edge2parallel_id;
  ForEachSuccCompTaskNode([&](CompTaskNode* succ_comp_node) {
    int64_t parallel_id = succ_comp_node->parallel_ctx()->parallel_id();
    min_out_parallel_id_ = std::min(min_out_parallel_id_, parallel_id);
    TaskNode* node = static_cast<TaskNode*>(succ_comp_node);
    while (true) {
      TaskNode* pred_node = node->SoleInEdge()->src_node();
      if (pred_node->task_id() == this->task_id()) {
        CHECK(edge2parallel_id.emplace(node->SoleInEdge(), parallel_id).second);
        return;
      } else {
        node = pred_node;
      }
    }
  });
  for (auto& pair : edge2parallel_id) {
    std::string regst_name = "out_" + std::to_string(pair.second - min_out_parallel_id_);
    pair.first->AddRegst(regst_name, ProduceRegst(regst_name));
  }
  ProduceRegst("data_tmp", 1, 1);
}

void ReduceLocalAddCompTaskNode::ConsumeAllRegsts() {
  int32_t in_regst_idx = 0;
  for (TaskEdge* edge : in_edges()) {
    ConsumeRegst("in_" + std::to_string(in_regst_idx), edge->GetSoleRegst());
    ++in_regst_idx;
  }
}

void ReduceLocalAddCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_local_add_conf;
  reduce_local_add_conf.set_name("reduce_local_add_" + NewUniqueId());
  reduce_local_add_conf.set_device_type(this->device_type());
  ReduceLocalAddOpConf* mut_local_add_conf = reduce_local_add_conf.mutable_reduce_local_add_conf();
  mut_local_add_conf->set_in_num(consumed_regsts().size());
  mut_local_add_conf->set_out_num(produced_regsts().size());
  mut_local_add_conf->set_first_parallel_id(min_out_parallel_id_);
  TaskNode* pred_task_node = (*out_edges().begin())->src_node();
  while (pred_task_node->GetTaskType() != TaskType::kReduceScatter) {
    pred_task_node = pred_task_node->SoleInEdge()->src_node();
  }
  std::shared_ptr<RegstDesc> diff_acc_regst =
      pred_task_node->consumed_regsts().begin()->second.front().lock();
  mut_local_add_conf->set_model_elem_cnt(diff_acc_regst->GetSoleBlobDesc()->shape().elem_cnt());
  std::shared_ptr<Operator> reduce_local_add_op = ConstructOp(reduce_local_add_conf);
  node->mut_op() = reduce_local_add_op;
  FOR_RANGE(size_t, i, 0, reduce_local_add_op->input_bns().size()) {
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in_" + std::to_string(i));
    node->BindBnWithRegst(reduce_local_add_op->input_bns().Get(i), in_regst);
  }
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, GetProducedRegst("data_tmp"));
  FOR_RANGE(size_t, i, 0, reduce_local_add_op->output_bns().size()) {
    std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out_" + std::to_string(i));
    out_regst->AddLbi(reduce_local_add_op->BnInOp2Lbi(reduce_local_add_op->output_bns().Get(i)));
    node->BindBnWithRegst(reduce_local_add_op->output_bns().Get(i), out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
