#include "oneflow/core/graph/reduce_gather2_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void ReduceGather2CompTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
}

void ReduceGather2CompTaskNode::ConsumeAllRegsts() {
  int64_t machine_num = logical_node()->parallel_desc()->sorted_machine_ids().size();
  int64_t dev_num_of_each_machine = logical_node()->parallel_desc()->device_num_of_each_machine();
  CHECK_EQ(machine_num * dev_num_of_each_machine, parallel_ctx()->parallel_num());
  bool do_local_reduce_scatter = machine_num > 1 && dev_num_of_each_machine > 1;

  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    while (dynamic_cast<CompTaskNode*>(src_node) == nullptr) {
      src_node = src_node->SoleInEdge()->src_node();
    }
    bool is_local_gather = src_node->GetTaskType() == TaskType::kReduceGather2;
    int64_t parallel_id = src_node->parallel_ctx()->parallel_id();

    int64_t in_edge_index = do_local_reduce_scatter
                                ? (is_local_gather ? parallel_id % dev_num_of_each_machine
                                                   : parallel_id / dev_num_of_each_machine)
                                : parallel_id;

    ConsumeRegst("in_" + std::to_string(in_edge_index), edge->GetSoleRegst());
  }
}

void ReduceGather2CompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  OperatorConf reduce_gather2_op_conf;
  reduce_gather2_op_conf.set_name("reduce_gather2_" + NewUniqueId());
  reduce_gather2_op_conf.set_device_type(this->device_type());
  reduce_gather2_op_conf.mutable_reduce_gather2_conf()->set_in_num(in_edges().size());
  std::shared_ptr<Operator> reduce_gather2_op = ConstructOp(reduce_gather2_op_conf);
  node->mut_op() = reduce_gather2_op;

  FOR_RANGE(size_t, i, 0, reduce_gather2_op->input_bns().size()) {
    node->BindBnWithRegst(reduce_gather2_op->input_bns().Get(i),
                          GetSoleConsumedRegst("in_" + std::to_string(i)));
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(reduce_gather2_op->BnInOp2Lbi(reduce_gather2_op->SoleObn()));
  node->BindBnWithRegst(reduce_gather2_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow
