#include "graph/exec_graph.h"

namespace oneflow {

void ExecEdge::set_lbn(const std::string& lbn) {
  lbn_ = lbn;
}

} // namespace oneflow
