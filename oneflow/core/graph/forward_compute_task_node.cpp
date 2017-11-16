#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/model_update_compute_task_node.h"

namespace oneflow {

void ForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto out_regst = ProduceRegst("out", 1, kMaxRegisterNum);
  auto activation_regst = ProduceRegst("activation", 1, kMaxRegisterNum);
  auto data_tmp_regst = ProduceRegst("data_tmp", 1, kMaxRegisterNum);

  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (dynamic_cast<BackwardCompTaskNode*>(dst_node)) {
      edge->AddRegst("activation", activation_regst);
      edge->AddRegst("data_tmp", data_tmp_regst);
    }
    edge->AddRegst("out", out_regst);
  }
}

void ForwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    if (dynamic_cast<MdUpdtCompTaskNode*>(src_node)) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else {
      ConsumeRegst("in", edge->GetRegst("out"));
    }
  }
}

void ForwardCompTaskNode::BuildExecGphAndRegst() {
  // TODO
}

void ForwardCompTaskNode::LockRegsts() {
  // TODO
}

bool ForwardCompTaskNode::IsReadyForBuild() {
  // TODO
  return false;
}

}  // namespace oneflow
