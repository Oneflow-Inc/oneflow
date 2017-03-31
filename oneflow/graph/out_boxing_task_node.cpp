#include "graph/out_boxing_task_node.h"

namespace oneflow {

// In future, we can use template-pattern
void OutBoxingTaskNode::FwBuildExecGraphAndSetProducedRegisterDescs() {
  SetOutEdgeRegisterPtr();
  Chain2EdgesMap chain2sorted_out_edges;
  FwInitChain2SortedEdgesMaps(&chain2sorted_out_edges,
                              &TaskNode::out_edges,
                              &TaskEdge::dst_node,
                              &TaskNode::SoleOutEdge);
  ChainEdgesPair chain_sorted_in_edges;
  chain_sorted_in_edges.first = chain_node();
  chain_sorted_in_edges.second.assign(in_edges().begin(), in_edges().end());
  FwSortEdgesInnerStage(&chain_sorted_in_edges,
                        &TaskEdge::src_node,
                        &TaskNode::SoleInEdge);
  for (const ChainEdgesPair& chain_sorted_out_edges : chain2sorted_out_edges) {
    FwBuildChainSortedEdgesPair(chain_sorted_in_edges, chain_sorted_out_edges);
  }
  SetProducedRegister();
  mut_exec_graph().UpdateSourceAndSink();
}

} // namespace oneflow
