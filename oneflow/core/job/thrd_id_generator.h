
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

class ThrdIdGenerator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThrdIdGenerator);

  ThrdIdGenerator(HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num)
      : machine_task_type2thrd_num_(machine_task_type2thrd_num) {
    for (auto pair : machine_task_type2thrd_num_) { InitThrdIdPool(pair.first, pair.second); }
  }

  int64_t GenerateThrdId(int64_t machine_id, int64_t task_type);

 private:
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

  int32_t GetPreThrdId(int64_t machine_id, int64_t task_type);

  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num_;
  HashMap<std::pair<int64_t, int64_t>, bool> machine_task_type2assigned_;
  HashMap<std::pair<int64_t, int64_t>, std::vector<int64_t>> machine_task_type2thrd_id_pool_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
