#include "graph/boxing_transformer_graph.h"

namespace oneflow {

void BoxingTransfmGraph::FwBuildGraph() {
  auto boxing_task_node = of_dynamic_cast<const BoxingTaskNode*>(task_node());
  // TODO
}

} // namespace oneflow
