#include "graph/in_boxing_task_node.h"

namespace oneflow {

void InBoxingTaskNode::FwBuildExecGraph() {
  Chain2EdgesMap chain2sorted_in_edges;
  FwInitChain2SortedEdgesMaps(&chain2sorted_in_edges, 
                              &TaskNode::in_edges,
                              &TaskEdge::src_node,
                              &TaskNode::SoleInEdge);
  ChainEdgesPair chain_sorted_out_edges;
  chain_sorted_out_edges.first = chain_node();
  chain_sorted_out_edges.second.assign(out_edges().begin(), out_edges().end());
  FwSortEdgesInnerStage(&chain_sorted_out_edges.second,
                        &TaskEdge::dst_node,
                        &TaskNode::SoleOutEdge);
  for (const ChainEdgesPair& chain_sorted_in_edges : chain2sorted_in_edges) {
    FwBuildChainSortedEdgesPair(chain_sorted_in_edges, chain_sorted_out_edges);
  }
  mut_exec_graph().UpdateSourceAndSink();
}

} // namespace oneflow
