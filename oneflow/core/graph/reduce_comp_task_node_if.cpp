#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"
#include "task_node.h"
#include "reduce_comp_task_node_if.h"

namespace oneflow {

int64_t InferRegstSize(const RegstDesc& regst) {
  return RtBlobDesc(*(regst.GetBlobDesc(GenPackedLbi()))).AlignedByteSizeOfBlobBody();
}

TaskNode* ReduceCompTaskNodeIf::FindPredReduceTaskNodeIf(std::function<bool(TaskNode*)> predicate) {
  TaskNode* current = AsCompTaskNode();
  while (current) {
    auto reduce_task_node_edge_it =
        std::find_if(current->in_edges().begin(), current->in_edges().end(), [](TaskEdge* edge) {
          return dynamic_cast<ReduceCompTaskNodeIf*>(edge->src_node()) != nullptr;
        });
    if (reduce_task_node_edge_it == current->in_edges().end()) { return nullptr; }
    current = (*reduce_task_node_edge_it)->src_node();
    if (predicate(current)) { return current; }
  }
  return nullptr;
}

}  // namespace oneflow
