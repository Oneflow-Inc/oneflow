#include "dag/segment_task_map.h"
#include <glog/logging.h>

namespace oneflow {

void SegmentTaskMap::AddSegmentForwardTaskPair(const std::string& segment_name,
  int32_t device_id, const std::string& forward_task) {
  auto task_pair_it = segment_to_forward_task_pair_.find(segment_name);
  if (task_pair_it == segment_to_forward_task_pair_.end()) {
    std::unordered_map<int32_t, std::string> task_pair;
    task_pair.insert({ device_id, forward_task });
    segment_to_forward_task_pair_.insert({ segment_name, task_pair });
  } else {
    auto task_it = task_pair_it->second.find(device_id);
    CHECK(task_it == task_pair_it->second.end());
    task_pair_it->second.insert({ device_id, forward_task });
  }
  return;
}
void SegmentTaskMap::AddSegmentBackwardTaskPair(const std::string& segment_name,
  int32_t device_id, const std::string& backward_task) {
  auto task_pair_it = segment_to_backward_task_pair_.find(segment_name);
  if (task_pair_it == segment_to_backward_task_pair_.end()) {
    std::unordered_map<int32_t, std::string> task_pair;
    task_pair.insert({ device_id, backward_task });
    segment_to_backward_task_pair_.insert({ segment_name, task_pair });
  } else {
    auto task_it = task_pair_it->second.find(device_id);
    CHECK(task_it == task_pair_it->second.end());
    task_pair_it->second.insert({ device_id, backward_task });
  }
  return;
}

std::string SegmentTaskMap::GetForwardTask(
  const std::string& segment_name, int32_t device_id) const {
  // TODO(Chonglin): to implement
  return "";
}

std::string SegmentTaskMap::GetBackwardTask(
  const std::string& segment_name, int32_t device_id) const {
  // TODO(Chonglin): to implement
  return "";
}

std::vector<int32_t> SegmentTaskMap::GetDeviceIDs(
  const std::string& segment_name) const {
  // TODO(jiyuan): to implement
  return {0};
}



}  // namespace oneflow