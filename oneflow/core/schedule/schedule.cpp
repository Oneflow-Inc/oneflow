#include "oneflow/core/schedule/schedule.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace schedule {

void Schedule::PrintRegstNum() {
  sgraph().ForEachRegstDesc([&](SRegstDesc* regst_desc) {
    float duration =
        GetOrDefault(regst_desc2duration(), regst_desc, static_cast<float>(0));
    float interval = max_interval();
    uint32_t count = GetOrDefault(regst_desc2count(), regst_desc, 1u);
    std::cout << "Allocation\t" << std::setw(15)
              << regst_desc->owner_task().id() << "\t" << std::setw(5)
              << regst_desc->id() << "\t" << count << "\t" << duration << ","
              << interval << std::endl;
  });
}

void Schedule::PrintSchedule() {
  auto batches = session().GetBatchNodes();
  std::cout << std::setw(15) << ""
            << " ";
  for (Batch* batch : *batches) {
    std::cout << std::setw(4) << batch->id() << " ";
  }
  std::cout << std::endl;
  sgraph().Walk([&](STask* task) {
    std::cout << std::setw(15) << task->id() << " ";
    for (Batch* batch : *batches) {
      TaskInstance* instance = session().task_instance_mgr().Find(batch, task);
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
  sgraph().ForEachRegstDesc([&](SRegstDesc* regst_desc) {
    STask* owner = nullptr;
    sgraph().produced_regst_desc_mgr().Input(regst_desc, &owner);
    uint32_t start = session().nr_unstable_batch();
    uint32_t end = start + session().nr_stable_batch();
    CHECK(end - start > 0);
    float sum = 0;
    for (uint32_t i = start; i < end; i++) {
      Batch* batch = session().batch_node_mgr().Find(i);
      TaskInstance* owner_instance =
          session().task_instance_mgr().Find(batch, owner);
      float duration = 0;
      sgraph().subscribed_regst_desc_mgr().Input(regst_desc, [&](STask* node) {
        TaskInstance* node_instance =
            session().task_instance_mgr().Find(batch, node);
        float d = GetDuration(owner_instance, node_instance);
        duration = std::max(duration, d);
      });
      sum += duration;
    }
    CHECK(end + 1 - start > 0);
    mut_regst_desc2duration()[regst_desc] = sum / (end - start);
  });
}

void Schedule::UpdateRegstCount() {
  sgraph().ForEachRegstDesc([&](SRegstDesc* regst_desc) {
    STask* owner = nullptr;
    sgraph().produced_regst_desc_mgr().Input(regst_desc, &owner);
    float duration =
        GetOrDefault(regst_desc2duration(), regst_desc, static_cast<float>(0));
    float interval = max_interval();
    float ratio = duration / std::max(interval, 1.0f);
    uint32_t count = ceil(ratio);
    if ((ratio - floor(ratio)) * 50 < interval) { count = floor(ratio); }
    mut_regst_desc2count()[regst_desc] = std::max(count, 1u);
  });
}

void Schedule::UpdateInterval() {
  STask* end_node = sgraph().sink();
  std::pair<float, float> default_range;
  float last_batch_ended_at = 0;
  std::vector<float> intervals;
  for (uint32_t i = 0; i < session().nr_batch(); ++i) {
    Batch* batch = session().batch_node_mgr().Find(i);
    TaskInstance* instance =
        session().task_instance_mgr().Find(batch, end_node);
    float current =
        GetOrDefault(instance2ended_at(), instance, default_range).second;
    if (i) { intervals.push_back(current - last_batch_ended_at); }
    last_batch_ended_at = current;
  }

  mut_max_interval() = GetInitiationIntervalFromIntervals(intervals);
}

float Schedule::GetInitiationIntervalFromIntervals(
    const std::vector<float>& intervals) {
  // guassion convoluted intervals
  std::vector<float> gci;
  uint32_t radius = intervals.size() / 6;
  for (int middle = intervals.size() / 4;
       middle - radius >= 0 && middle + radius < intervals.size(); ++middle) {
    float sum = 0;
    for (int i = middle - radius; i <= middle + radius; ++i) {
      sum += intervals[i];
    }
    float x = sum / (radius * 2 + 1);
    gci.push_back(x);
  }

  // drop the outlier
  std::sort(gci.begin(), gci.end(), std::less<float>());
  float sum = 0;
  uint32_t count = 0;
  uint32_t start_margin = gci.size() / 5;
  uint32_t end_margin = gci.size() / 5;
  for (int i = start_margin; i < gci.size() - end_margin; ++i) {
    float x = gci[i];
    sum += x;
    ++count;
  }

  return sum / count;
}

void Schedule::Clear() {
  mut_instance2ended_at().clear();
  mut_device2ended_at().clear();
  mut_regst_desc2duration().clear();
}

}  // namespace schedule
}  // namespace oneflow
