#include "oneflow/core/graph/node.h"

namespace oneflow {

int64_t NewNodeId() {
  static int64_t node_id = 0;
  return node_id++;
}

int64_t NewEdgeId() {
  static int64_t edge_id = 0;
  return edge_id++;
}

}  // namespace oneflow
