#ifndef _DAG_SEGMENT_TASK_MAP_H_
#define _DAG_SEGMENT_TASK_MAP_H_
#include <unordered_map>
#include <string>
#include <cstdint>
namespace oneflow {
// A compute segment may be instantiated on multiple devices. On each device, 
// there is a TaskDag corresponding to this segment.

class SegmentTaskMap {
public:
  SegmentTaskMap() = default;
  ~SegmentTaskMap() = default;

  void AddSegmentForwardTaskPair(const std::string& segment_name,
    int32_t device_id, const std::string& forward_task);
  void AddSegmentBackwardTaskPair(const std::string& segment_name,
    int32_t device_id, const std::string& backward_task);

  std::vector<int32_t> GetDeviceIDs(const std::string& segment_name) const;

  std::string GetForwardTask(
    const std::string& segment_name, int32_t device_id) const;
  std::string GetBackwardTask(
    const std::string& segment_name, int32_t device_id) const;

private:
  std::unordered_map<std::string, std::unordered_map<int32_t, std::string>>
    segment_to_forward_task_pair_;
  std::unordered_map<std::string, std::unordered_map<int32_t, std::string>>
    segment_to_backward_task_pair_;
};
}  // namespace oneflow
#endif  // _DAG_SEGMENT_TASK_MAP_H_