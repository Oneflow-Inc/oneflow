#include "oneflow/core/schedule/session.h"
namespace oneflow {
namespace schedule {

void Session::InitNodeBatchInstance(STask* node) {
  for (uint32_t i = 0; i < nr_batch(); i++) {
    auto batch = batch_node_mgr().Find(i);
    mut_task_instance_mgr().CreateIfNotFound(batch, node);
  }
}

void Session::NewBatchs() {
  std::list<Batch*> batch_nodes;
  for (int i = 0; i < nr_batch(); i++) {
    auto batch = mut_batch_node_mgr().CreateWithId(i, std::to_string(i));
    batch_nodes.push_back(batch);
  }
  graph()->ForeachNodeWithSourceAndSink([&](STask* node) {
    for (auto batch : batch_nodes) {
      mut_task_instance_mgr().CreateIfNotFound(batch, node);
    }
  });
  graph()->ForeachArc([&](TaskArc* arc) {
    for (auto batch : batch_nodes) {
      mut_task_arc_instance_mgr().CreateIfNotFound(batch, arc);
    }
  });
  graph()->ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    for (auto batch : batch_nodes) {
      mut_regst_desc_instance_mgr().CreateIfNotFound(batch, regst_desc);
    }
  });
}

std::unique_ptr<std::list<Batch*>> Session::GetBatchNodes() {
  auto batchs = unique_ptr_new<std::list<Batch*>>();
  for (uint32_t i = 0; i < nr_batch(); i++) {
    batchs->push_back(batch_node_mgr().Find(i));
  }
  return batchs;
}

}  // namespace schedule
}  // namespace oneflow
