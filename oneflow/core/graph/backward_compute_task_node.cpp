#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/model_diff_accumulate_compute_task_node.h"
#include "oneflow/core/graph/model_update_compute_task_node.h"

namespace oneflow {

void BackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  bool need_in_diff = false;
  chain_node()->ForEachNodeOnOutEdge([&](const ChainNode* out_node) {
    if (dynamic_cast<const BackwardChainNode*>(out_node)) {
      need_in_diff = true;
    }
  });
  if (need_in_diff) {
    auto in_diff_regst = ProduceRegst("in_diff", 1, kMaxRegisterNum);
    for (TaskEdge* edge : out_edges()) {
      TaskNode* dst_node = edge->dst_node();
      if (!dynamic_cast<MdDiffAccCompTaskNode*>(dst_node)) {
        edge->AddRegst("in_diff", in_diff_regst);
      }
    }
  }
  auto model_diff_regst = ProduceRegst("model_diff", 1, kMaxRegisterNum);
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (dynamic_cast<MdDiffAccCompTaskNode*>(dst_node)) {
      edge->AddRegst("model_diff", model_diff_regst);
    }
  }
  ProduceRegst("activation_diff", 1, 1);
}

void BackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    if (dynamic_cast<ForwardCompTaskNode*>(src_node)) {
      ConsumeRegst("activation", edge->GetRegst("activation"));
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("out", edge->GetRegst("out"));
    } else if (dynamic_cast<MdUpdtCompTaskNode*>(src_node)) {
      ConsumeRegst("model", edge->GetRegst("model"));
    } else if (dynamic_cast<BoxingTaskNode*>(src_node)) {
      ConsumeRegst("out_diff", edge->GetRegst("out"));
    }
    // boxing node may be deleted
    else {
      ConsumeRegst("out_diff", edge->GetRegst("in_diff"));
    }
  }
}

void BackwardCompTaskNode::BuildExecGphAndRegst() {
  // TODO
}

void BackwardCompTaskNode::LockRegsts() {
  // TODO
}

bool BackwardCompTaskNode::IsReadyForBuild() {
  // TODO
  return false;
}

}  // namespace oneflow
