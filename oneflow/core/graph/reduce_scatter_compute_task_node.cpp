#include "oneflow/core/graph/reduce_scatter_compute_task_node.h"

namespace oneflow {

void ReduceScatterCompTaskNode::ProduceAllRegstsAndBindEdges() {
  int64_t index = 0;
  for (TaskEdge* edge : out_edges()) {
    std::string out_regst_name = "out_" + std::to_string(index);
    edge->AddRegst(out_regst_name, ProduceRegst(out_regst_name));
    ++index;
  }
}

void ReduceScatterCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", this->SoleInEdge()->GetSoleRegst());
}

void ReduceScatterCompTaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  // TODO
}

void ReduceScatterCompTaskNode::BuildExecGphAndRegst() {
  // TODO
}

}  // namespace oneflow
