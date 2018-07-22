
#ifndef ONEFLOW_THRD_ID_DISTRIBUTOR_H
#define ONEFLOW_THRD_ID_DISTRIBUTOR_H

#include <vector>
#include <map>
#include <array>

#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class ThrdIdDistributor {
 public:
  static ThrdIdDistributor& get() {
    static ThrdIdDistributor inst;
    return inst;
  }

  void AddThrdNum(TaskType task_type, int32_t thrd_num) {
    task_type2thrd_num_.emplace(task_type, thrd_num);
  }

  int64_t GenerateThrdId(TaskType task_type, int32_t offset);

  int64_t GetThrdId(TaskType task_type, int32_t index) const;

  int64_t GetThrdNum(TaskType task_type) {
    auto it = task_type2thrd_num_.find(task_type);
    return it->second;
  }

  std::vector<int64_t> GetThrdIds(TaskType task_type) const {
    auto it = task_type2thrd_ids_.find(task_type);
    if (it != task_type2thrd_ids_.end()) return it->second;

    return {};
  }

  const std::vector<TaskType> PersistenceThrdTypes() const {
    std::vector<TaskType> v;
    for (auto const& pair : task_type2assigned_) {
      if (pair.first != TaskType::kCopyCommNet) v.push_back(pair.first);
    }

    return v;
  }

 private:
  ThrdIdDistributor();

  ThrdIdDistributor(const ThrdIdDistributor&) = delete;
  ThrdIdDistributor(ThrdIdDistributor&&) = delete;

  int32_t GetPreThrdId(TaskType task_type);

  const JobDesc* desc_ = Global<JobDesc>::Get();
  const int64_t BASE_THRD_ID =
      desc_->resource().gpu_device_num() * 4 + desc_->resource().cpu_device_num();
  std::map<TaskType, int> task_type2thrd_num_;

  std::map<TaskType, bool> task_type2assigned_;

  std::map<TaskType, std::vector<int64_t>> task_type2thrd_ids_;
};
}  // namespace oneflow

#endif  // ONEFLOW_THRD_ID_DISTRIBUTOR_H
