#include "oneflow/core/graph/node.h"

namespace oneflow {

uint64_t NewNodeId() {
  static uint64_t node_id = 0;
  return node_id++;
}

uint64_t NewEdgeId() {
  static uint64_t edge_id = 0;
  return edge_id++;
}

} // namespace oneflow
