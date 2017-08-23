#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace schedule {

void Schedule::PrintRegstNum() {
  session()->graph()->ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    float duration =
        GetOrDefault(regst_desc2duration(), regst_desc, static_cast<float>(0));
    float interval = max_interval();
    uint32_t count = GetOrDefault(regst_desc2count(), regst_desc, 1u);
    std::cout << "Allocation\t" << regst_desc->owner_task()->id() << "\t"
              << regst_desc->id() << "\t" << count << "\t" << duration << ","
              << interval << std::endl;
  });
}

float Schedule::GetDuration(TaskInstance* src_node, TaskInstance* dst_node) {
  std::pair<float, float> default_pair;
  float end = GetOrDefault(instance2ended_at(), dst_node, default_pair).second;
  float start = GetOrDefault(instance2ended_at(), src_node, default_pair).first;
  return end - start;
}

void Schedule::UpdateDuration() {
  session()->graph()->ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    STask* owner = nullptr;
    session()->graph()->produced_regst_desc_mgr().Input(regst_desc, &owner);
    float duration = 0;
    uint32_t start = session()->nr_base_batch();
    uint32_t end = start + session()->nr_base_batch();
    session()->graph()->subscribed_regst_desc_mgr().Input(
        regst_desc, [&](STask* node) {
          float sum = 0;
          std::set<float> cases;
          for (uint32_t i = start; i < end; i++) {
            Batch* batch = session()->batch_node_mgr().Find(i);
            TaskInstance* owner_instance =
                session()->task_instance_mgr().Find(batch, owner);
            TaskInstance* node_instance =
                session()->task_instance_mgr().Find(batch, node);
            float d = GetDuration(owner_instance, node_instance);
            cases.insert(d);
          }
          CHECK(cases.size());
          for (float x : cases) { sum += x; }
          float avg = sum / cases.size();
          duration = std::max(duration, avg);
        });
    mut_regst_desc2duration()[regst_desc] = duration;
  });
}

void Schedule::UpdateRegstCount() {
  session()->graph()->ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    STask* owner = nullptr;
    session()->graph()->produced_regst_desc_mgr().Input(regst_desc, &owner);
    float duration =
        GetOrDefault(regst_desc2duration(), regst_desc, static_cast<float>(0));
    float interval = max_interval();
    uint32_t count = ceil(duration / std::max(interval, 1.0f));
    mut_regst_desc2count()[regst_desc] = std::max(count, 1u);
  });
}

void Schedule::UpdateInterval() {
  STask* end_node = session()->graph()->sink();
  float sum = 0.0;
  float last_time = 0.0;
  uint32_t start = session()->nr_base_batch();
  uint32_t end = start + session()->nr_base_batch();
  CHECK(end - start > 1);
  std::set<float> cases;
  std::pair<float, float> default_range;
  for (uint32_t i = start; i < end; i++) {
    Batch* batch = session()->batch_node_mgr().Find(i);
    TaskInstance* instance =
        session()->task_instance_mgr().Find(batch, end_node);
    float start_time =
        GetOrDefault(instance2ended_at(), instance, default_range).first;
    if (i > start) { cases.insert(start_time - last_time); }
    last_time = start_time;
  }
  for (float x : cases) { sum += x; }
  mut_max_interval() = sum / cases.size();
}

void Schedule::Clear() {
  mut_instance2ended_at().clear();
  mut_device2ended_at().clear();
  mut_regst_desc2duration().clear();
}

}  // namespace schedule
}  // namespace oneflow
