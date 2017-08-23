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
    std::cout << "Allocation\t" << std::setw(15)
              << regst_desc->owner_task()->id() << "\t" << std::setw(5)
              << regst_desc->id() << "\t" << count << "\t" << duration << ","
              << interval << std::endl;
  });
}

void Schedule::PrintSchedule() {
  auto batches = session()->GetBatchNodes();
  std::cout << std::setw(15) << ""
            << " ";
  for (Batch* batch : *batches) {
    std::cout << std::setw(4) << batch->id() << " ";
  }
  std::cout << std::endl;
  session()->graph()->Walk([&](STask* task) {
    std::cout << std::setw(15) << task->id() << " ";
    for (Batch* batch : *batches) {
      TaskInstance* instance = session()->task_instance_mgr().Find(batch, task);
      float start = instance2ended_at_[instance].first;
      std::cout << std::setw(4) << start << " ";
    }
    std::cout << std::endl;
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
    CHECK(end - start > 0);
    session()->graph()->subscribed_regst_desc_mgr().Input(
        regst_desc, [&](STask* node) {
          float sum = 0;
          for (uint32_t i = start; i < end; i++) {
            Batch* batch = session()->batch_node_mgr().Find(i);
            TaskInstance* owner_instance =
                session()->task_instance_mgr().Find(batch, owner);
            TaskInstance* node_instance =
                session()->task_instance_mgr().Find(batch, node);
            float d = GetDuration(owner_instance, node_instance);
            sum += d;
          }
          float avg = sum / (end - start);
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
  uint32_t start = session()->nr_base_batch();
  uint32_t end = start + session()->nr_base_batch();
  CHECK(end - start > 1);
  std::pair<float, float> default_range;
  std::list<float> start_time;
  for (uint32_t i = start; i < end; i += (end - 1 - start)) {
    Batch* batch = session()->batch_node_mgr().Find(i);
    TaskInstance* instance =
        session()->task_instance_mgr().Find(batch, end_node);
    float currrent_start_time =
        GetOrDefault(instance2ended_at(), instance, default_range).first;
    start_time.push_back(currrent_start_time);
  }
  CHECK(start_time.size());
  mut_max_interval() = (start_time.back() - start_time.front()) / (end - start);
}

void Schedule::Clear() {
  mut_instance2ended_at().clear();
  mut_device2ended_at().clear();
  mut_regst_desc2duration().clear();
}

}  // namespace schedule
}  // namespace oneflow
