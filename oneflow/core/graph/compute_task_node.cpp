#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void CompTaskNode::ToProto(TaskProto* task_proto) {
  TaskNode::ToProto(task_proto);
  *(task_proto->mutable_parallel_ctx()) = parallel_ctx_;
  task_proto->set_chain_id(chain_node()->node_id());
}

const ChainNode* CompTaskNode::SuccChainNodeOnEdge(TaskEdge* edge) {
  CompTaskNode* succ_comp_node = nullptr;
  do {
    TaskNode* dst_node = edge->dst_node();
    edge = *(dst_node->out_edges().begin());
    succ_comp_node = dynamic_cast<CompTaskNode*>(dst_node);
  } while (!succ_comp_node && edge);
  if (succ_comp_node) { return succ_comp_node->chain_node(); }
  return nullptr;
}

const ChainNode* CompTaskNode::PredChainNodeOnEdge(TaskEdge* edge) {
  CompTaskNode* pred_comp_node = nullptr;
  do {
    TaskNode* src_node = edge->src_node();
    edge = *(src_node->in_edges().begin());
    pred_comp_node = dynamic_cast<CompTaskNode*>(src_node);
  } while (!pred_comp_node && edge);
  if (pred_comp_node) { return pred_comp_node->chain_node(); }
  return nullptr;
}

void SortByParallelId(std::vector<CompTaskNode*>* node_vec) {
  std::sort(node_vec->begin(), node_vec->end(),
            [](const CompTaskNode* lhs, const CompTaskNode* rhs) {
              return lhs->parallel_ctx()->parallel_id()
                     < rhs->parallel_ctx()->parallel_id();
            });
}

}  // namespace oneflow
