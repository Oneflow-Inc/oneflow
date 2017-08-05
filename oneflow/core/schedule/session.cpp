#include "oneflow/core/schedule/session.h"
namespace oneflow {
namespace schedule {

void Session::InitNodeBatchInstance(Node* node) {
  for (uint32_t i = 0; i < nr_batch(); i++) {
    auto batch = batch_node_mgr().Find(i);
    mut_batch_arc_mgr().CreateIfNotFound(batch, node);
  }
}

void Session::NewBatchs() {
  std::list<Node*> batch_nodes;
  for (int i = 0; i < nr_batch(); i++) {
    auto batch = mut_batch_node_mgr().CreateWithId(i, std::to_string(i));
    batch_nodes.push_back(batch);
  }
  graph()->ForeachNodeWithSourceAndSink([&](Node* node) {
    for (auto batch : batch_nodes) {
      auto instance = mut_batch_arc_mgr().CreateIfNotFound(batch, node);
      mut_batch_instance_node_mgr().CreateWithId(
          instance->id(), std::to_string(instance->id()));
    }
  });
  graph()->ForeachArc([&](Arc* arc) {
    auto place = dynamic_cast<Node*>(arc);
    for (auto batch : batch_nodes) {
      mut_batch_arc_mgr().CreateIfNotFound(batch, place);
    }
  });
  graph()->ForeachRegstDesc([&](Node* regst_desc) {
    for (auto batch : batch_nodes) {
      mut_batch_arc_mgr().CreateIfNotFound(batch, regst_desc);
    }
  });
}

}  // namespace schedule
}  // namespace oneflow
