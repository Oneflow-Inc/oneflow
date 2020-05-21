#include "oneflow/core/graph/distribute_split_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

bool DistributeSplitCompTaskNode::HasBackwardCompTaskNode() { return false; }

void DistributeSplitCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", true);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void DistributeSplitCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

bool DistributeSplitCompTaskNode::IsReadyForBuild() {
  return GetSoleConsumedRegst("in")->IsLocked();
}

void DistributeSplitCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void DistributeSplitCompTaskNode::BuildExecGphStructAndBindInRegst() {
  for (std::shared_ptr<const Operator> op : logical_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
  }
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      cur_node->BindBnWithRegst(ibn, GetSoleConsumedRegst("in"));
    }
  });
}

void DistributeSplitCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    const auto& obn = cur_node->op()->output_bns().Get(parallel_ctx()->parallel_id());
    out_regst->AddLbi(cur_node->op()->BnInOp2Lbi(obn));
    cur_node->BindBnWithRegst(obn, out_regst);
  });
  // NOTE: we can ONLY set inplace when regst has ONLY ONE blob
  auto in_regst = GetSoleConsumedRegst("in");
  if (in_regst->NumOfLbi() == 1) {
    out_regst->set_hint_inplace_consumed_regst_desc_id(in_regst->regst_desc_id());
  }
}  // namespace oneflow

void DistributeSplitCompTaskNode::InferProducedDataRegstTimeShape() {
  *GetProducedRegst("out")->mut_data_regst_time_shape() =
      GetSoleConsumedRegst("in")->data_regst_time_shape();
}

}  // namespace oneflow
