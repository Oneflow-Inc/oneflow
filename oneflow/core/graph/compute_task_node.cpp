#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void CompTaskNode::ToProto(TaskProto* task_proto) {
  TaskNode::ToProto(task_proto);
  *(task_proto->mutable_parallel_ctx()) = parallel_ctx_;
  task_proto->set_chain_id(chain_node()->node_id());
}

bool CompTaskNode::IsRecurrentOutEdge(TaskEdge* edge) {
  CompTaskNode* self_node = dynamic_cast<CompTaskNode*>(edge->src_node());
  CHECK(self_node);
  CompTaskNode* next_comp_node = nullptr;
  do {
    TaskNode* dst_node = edge->dst_node();
    edge = *(dst_node->out_edges().begin());
    next_comp_node = dynamic_cast<CompTaskNode*>(dst_node);
  } while (!next_comp_node && edge);
  if (next_comp_node
      && next_comp_node->chain_node() == self_node->chain_node()) {
    return true;
  }
  return false;
}

void SortByParallelId(std::vector<CompTaskNode*>* node_vec) {
  std::sort(node_vec->begin(), node_vec->end(),
            [](const CompTaskNode* lhs, const CompTaskNode* rhs) {
              return lhs->parallel_ctx()->parallel_id()
                     < rhs->parallel_ctx()->parallel_id();
            });
}

}  // namespace oneflow
