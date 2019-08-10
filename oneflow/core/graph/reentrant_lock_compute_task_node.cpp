#include "oneflow/core/graph/reentrant_lock_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

void ReentrantLockCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void ReentrantLockCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in");
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void ReentrantLockCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  // no regst_desc for ibn "end" provided because TaskGraph hates cycle
  node->BindBnWithOneOfTheRegsts("start", in_regsts);
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    out_regst->AddLbi(lbi);
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void ReentrantLockCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<Shape> time_shape(new Shape());
  for (TaskEdge* edge : in_edges()) {
    if (edge->src_node()->GetFastestInputOutputTimeShape()) {
      *time_shape = *edge->src_node()->GetFastestInputOutputTimeShape();
    }
  }
  CHECK_GT(time_shape->elem_cnt(), 0);
  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

REGISTER_TICK_TOCK_TASK_TYPE(TaskType::kReentrantLock);

}  // namespace oneflow
