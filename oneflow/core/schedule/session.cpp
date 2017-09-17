#include "oneflow/core/schedule/session.h"
namespace oneflow {
namespace schedule {

void Session::NewBatchs() {
  for (uint32_t i = 0u; i < nr_batch(); i++) {
    const Batch* batch =
        mut_batch_node_mgr()->CreateWithId(i, std::to_string(i));
    sgraph().ForEachNode([&](const STask* node) {
      mut_task_instance_mgr()->CreateIfNotFound(batch, node);
    });
    sgraph().ForEachArc([&](const TaskArc* arc) {
      mut_task_arc_instance_mgr()->CreateIfNotFound(batch, arc);
    });
    sgraph().ForEachRegstDesc([&](const SRegstDesc* regst_desc) {
      mut_regst_desc_instance_mgr()->CreateIfNotFound(batch, regst_desc);
    });
  }
}

std::unique_ptr<std::list<const Batch*>> Session::GetBatchNodes() const {
  auto batchs = of_make_unique<std::list<const Batch*>>();
  for (uint32_t i = 0u; i < nr_batch(); i++) {
    batchs->push_back(batch_node_mgr().Find(i));
  }
  return batchs;
}

const TaskInstance* Session::GetPrevBatchInstance(
    const TaskInstance* instance) const {
  return GetNextBatchInstance(instance, static_cast<int32_t>(-1));
}  // namespace schedule

const TaskInstance* Session::GetNextBatchInstance(const TaskInstance* instance,
                                                  int32_t step) const {
  const TaskInstance* next = nullptr;
  const Batch* batch = instance->src_node();
  uint32_t next_batch_id = batch->id() + step;
  const Batch* next_batch = batch_node_mgr().Find(next_batch_id);
  next = task_instance_mgr().Find(next_batch, instance->dst_node());
  return next;
}

}  // namespace schedule
}  // namespace oneflow
