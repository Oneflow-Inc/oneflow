#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/chain_graph.h"

namespace oneflow {

void BackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  bool need_in_diff = false;
  chain_node()->ForEachNodeOnOutEdge([&](const ChainNode* out_node) {
    if (dynamic_cast<const BackwardChainNode*>(out_node)) {
      need_in_diff = true;
    }
  });
  if (need_in_diff) { ProduceRegst("in_diff", 1, kMaxRegisterNum); }
  ProduceRegst("model_diff", 1, kMaxRegisterNum);
  ProduceRegst("activation_diff", 1, 1);
}

}  // namespace oneflow
