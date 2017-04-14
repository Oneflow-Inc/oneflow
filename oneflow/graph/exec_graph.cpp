#include "graph/exec_graph.h"

namespace oneflow {

void ExecEdge::set_lbn(const std::string& lbn) {
  lbn_ = lbn;
  pbn_ = "edge_" + std::to_string(edge_id()) + "/" + lbn;
}

} // namespace oneflow
