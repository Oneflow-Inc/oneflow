#include "graph/exec_graph.h"
#include "graph/task_node.h"

namespace oneflow {

void ExecGraph::BuildGraph() {
  if (task_node_->IsFwNode()) {
    FwBuildGraph();
  } else {
    BpBuildGraph();
  }
}

void ExecGraph::SubscribeRegisterDescInnerPath() {
  for (TaskEdge* edge : task_node()->in_edges()) {
    CHECK_NOTNULL(edge->register_desc());
    edge->register_desc()->AddSubscriber(this);
    subscribed_register_descs_.insert(edge->register_desc());
  }
}

} // namespace oneflow
