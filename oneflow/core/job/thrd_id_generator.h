
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

  ThrdIdGenerator(HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num,
                  int64_t base_thrd_id)
      : machine_task_type2thrd_num_(machine_task_type2thrd_num), base_thrd_id_(base_thrd_id) {
    machine_task_type2lowerbound_.resize(machine_task_type2thrd_num.size());

    for (auto pair : machine_task_type2thrd_num_) {
      int64_t task_type = pair.first.second;
      AdjustThrdNum(task_type, pair.second);
      InitThrdIdPool(pair.first, pair.second);
    }
  }

  int64_t GenerateThrdId(int64_t machine_id, int64_t task_type);

  static bool IsPesistence(int64_t task_type) {
    bool is_persistence = (task_type == TaskType::kRecordLoad || task_type == TaskType::kLossPrint
                           || task_type == TaskType::kMdSave || task_type == TaskType::kPrint
                           || task_type == TaskType::kAccuracyPrint);
    return is_persistence;
  }

 private:
  void AdjustThrdNum(int64_t task_type, int32_t& thrd_num) {
    CHECK(IsPesistence(task_type));
    if (task_type == TaskType::kMdSave) {
      JobDesc* job_desc = Global<JobDesc>::Get();
      const int32_t mdsave_conf_num = job_desc ? job_desc->MdSaveWorkerNum() : 64;
      thrd_num = thrd_num > mdsave_conf_num ? mdsave_conf_num : thrd_num;
    }
  }

  void InitThrdIdPool(std::pair<int64_t, int64_t> machine_task_type, int32_t thrd_num) {
    std::vector<int64_t> thrd_id_pool(thrd_num);
    std::iota(thrd_id_pool.begin(), thrd_id_pool.end(), 0);
    std::reverse(thrd_id_pool.begin(), thrd_id_pool.end());  // such as [3,2,1,0]
    machine_task_type2thrd_id_pool_[machine_task_type] = std::move(thrd_id_pool);
  }

  void InitThrdIdPool(std::pair<int64_t, int64_t> machine_task_type) {
    int32_t thrd_num = GetThrdNum(machine_task_type);

    InitThrdIdPool(machine_task_type, thrd_num);
  }

  int64_t GetThrdIdFromPool(std::pair<int64_t, int64_t> machine_task_type) {
    if (machine_task_type2thrd_id_pool_[machine_task_type].empty()) {
      InitThrdIdPool(machine_task_type);
    }

    auto& thrd_ids_in_pool = machine_task_type2thrd_id_pool_[machine_task_type];
    int64_t thrd_id_from_pool = thrd_ids_in_pool.back();
    thrd_ids_in_pool.pop_back();
    return thrd_id_from_pool;
  }

  int64_t GetThrdNum(std::pair<int64_t, int64_t> machine_task_type) {
    auto it = machine_task_type2thrd_num_.find(machine_task_type);
    CHECK(it != machine_task_type2thrd_num_.end());
    return it->second;
  }

  int64_t GetPreThrdId(int64_t machine_id, int64_t task_type);

  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num_;
  int64_t base_thrd_id_;
  std::vector<HashMap<int64_t, int32_t>> machine_task_type2lowerbound_;
  HashMap<std::pair<int64_t, int64_t>, std::vector<int64_t>> machine_task_type2thrd_id_pool_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
