#include "graph/transfm_graph.h"
#include "graph/task_node.h"

namespace oneflow {

void TransfmGraph::BuildGraph() {
  if (task_node_->IsFwNode()) {
    FwBuildGraph();
  } else {
    BpBuildGraph();
  }
}

void TransfmGraph::SubscribeRegisterDescInnerPath() {
  for (TaskEdge* edge : task_node()->in_edges()) {
    CHECK_NOTNULL(edge->register_desc());
    edge->register_desc()->AddSubscriber(this);
    subscribed_register_descs_.insert(edge->register_desc());
  }
}

} // namespace oneflow
