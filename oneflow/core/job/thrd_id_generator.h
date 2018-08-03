
#ifndef ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
#define ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_

#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/job/job_desc.h"

namespace std {

template<>
struct hash<std::pair<int64_t, int64_t>> {
  size_t operator()(const std::pair<int64_t, int64_t>& pair) const {
    return std::hash<size_t>()(pair.first) ^ std::hash<size_t>()(pair.second);
  }
};

}  // namespace std

namespace oneflow {

class ThrdIdGenerator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThrdIdGenerator);

  ThrdIdGenerator(std::vector<std::pair<int64_t, TaskType>>& machine_task_type_vec,
                  int64_t base_thrd_id);

  int64_t GenerateThrdId(int64_t machine_id, int64_t task_type) {
    auto key = std::make_pair(machine_id, task_type);
    return machine_task_type2lowerbound_[key] + GetThrdIdFromPool(key);
  }

 private:
  void InitLowerboundOfTaskType(
      const HashMap<int64_t, std::vector<TaskType>>& machine2task_type_seq);

  int64_t GetThrdIdFromPool(std::pair<int64_t, int64_t> machine_task_type) {
    if (machine_task_type2thrd_id_pool_[machine_task_type].empty()) {
      InitThrdIdPool(machine_task_type);
    }

    auto& thrd_ids_in_pool = machine_task_type2thrd_id_pool_[machine_task_type];
    int64_t thrd_id_from_pool = thrd_ids_in_pool.back();
    thrd_ids_in_pool.pop_back();
    return thrd_id_from_pool;
  }

  void InitThrdIdPool(std::pair<int64_t, int64_t> machine_task_type) {
    int32_t thrd_num = machine_task_type2thrd_num_[machine_task_type];

    std::vector<int64_t> thrd_id_pool(thrd_num);
    std::iota(thrd_id_pool.begin(), thrd_id_pool.end(), 0);
    std::reverse(thrd_id_pool.begin(), thrd_id_pool.end());  // such as [3,2,1,0]
    machine_task_type2thrd_id_pool_[machine_task_type] = std::move(thrd_id_pool);
  }

  template<typename T>
  size_t Unique(std::vector<T>& vec) {
    std::set<T> temp;

    auto removed_start = std::remove_if(vec.begin(), vec.end(), [&temp](const T& value) {
      if (temp.find(value) != std::end(temp)) return true;

      temp.insert(value);
      return false;
    });

    vec.erase(removed_start, vec.end());

    return vec.size();
  }

  bool EqualConf(int64_t task_type, int32_t thrd_num) {
    if (task_type == TaskType::kMdSave) {
      JobDesc* job_desc = Global<JobDesc>::Get();
      const int32_t mdsave_conf_num = job_desc ? job_desc->MdSaveWorkerNum() : 4;
      if (thrd_num == mdsave_conf_num) return true;
    }

    return false;
  }

  int64_t base_thrd_id_;
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num_;
  HashMap<std::pair<int64_t, int64_t>, std::vector<int64_t>> machine_task_type2thrd_id_pool_;
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2lowerbound_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
