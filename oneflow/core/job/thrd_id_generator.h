#ifndef ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
#define ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_

#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class ThrdIdGenerator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThrdIdGenerator);

  ThrdIdGenerator(std::vector<std::pair<int64_t, TaskType>>& machine_task_type_vec,
                  int64_t base_thrd_id);

  int64_t GenerateThrdId(int64_t machine_id, int64_t task_type) {
    auto key = std::make_pair(machine_id, task_type);
    int64_t ret = machine_task_type2lowerbound_[key] + GetModThrdId(key);
    CHECK_GE(ret, 0);
    return ret;
  }

 private:
  void InitLowerboundOfTaskType(const HashMap<int64_t, std::set<TaskType>>& machine2task_types);

  int64_t GetModThrdId(std::pair<int64_t, int64_t> machine_task_type) {
    int64_t& offset = machine_task_type2offset_[machine_task_type];
    int64_t mod_thrd_id = offset % machine_task_type2thrd_num_[machine_task_type];
    offset++;
    return mod_thrd_id;
  }

  bool EqualConf(int64_t task_type, int32_t thrd_num) {
    if (task_type == TaskType::kMdSave) {
      if (thrd_num == Global<ResourceDesc>::Get()->MaxMdSaveWorkerNum()) { return true; }
    }

    return false;
  }

  int64_t base_thrd_id_;
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num_;
  HashMap<std::pair<int64_t, int64_t>, int64_t> machine_task_type2offset_;
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2lowerbound_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
