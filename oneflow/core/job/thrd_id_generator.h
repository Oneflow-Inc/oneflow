
#ifndef ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
#define ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_

#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class ThrdIdGenerator {
 public:
  static ThrdIdGenerator& get() {
    static ThrdIdGenerator inst;
    return inst;
  }

  void AddThrdNum(std::pair<int32_t, int32_t> machine_task_type, int32_t thrd_num) {
    machine_task_type2thrd_num_.emplace(machine_task_type, thrd_num);

    GenerateThrdIdPool(machine_task_type, thrd_num);
  }

  int64_t GenerateThrdId(int32_t machine_id, int32_t task_type);

  bool IsPersistence(std::pair<int32_t, int32_t> machine_task_type) {
    return machine_task_type2thrd_num_.find(machine_task_type) != machine_task_type2thrd_num_.end();
  }

  const HashMap<std::pair<int32_t, int32_t>, std::vector<int64_t>>& GetThrdIds() const {
    return machine_task_type2thrd_ids_;
  }

 private:
  ThrdIdGenerator() {}
  OF_DISALLOW_COPY_AND_MOVE(ThrdIdGenerator);

  void GenerateThrdIdPool(std::pair<int32_t, int32_t> machine_task_type) {
    int32_t thrd_num = GetThrdNum(machine_task_type);

    GenerateThrdIdPool(machine_task_type, thrd_num);
  }

  void GenerateThrdIdPool(std::pair<int32_t, int32_t> machine_task_type, int32_t thrd_num) {
    std::vector<int64_t> thrd_id_pool(thrd_num);
    std::iota(thrd_id_pool.begin(), thrd_id_pool.end(), 0);
    std::reverse(thrd_id_pool.begin(), thrd_id_pool.end());  // such as [3,2,1,0]
    machine_task_type2thrd_id_pool_[machine_task_type] = std::move(thrd_id_pool);
  }

  int64_t GetThrdIdFromPool(std::pair<int32_t, int32_t> machine_task_type) {
    if (machine_task_type2thrd_id_pool_[machine_task_type].empty()) {
      GenerateThrdIdPool(machine_task_type);
    }

    auto& thrd_ids_in_pool = machine_task_type2thrd_id_pool_[machine_task_type];
    int64_t thrd_id_from_pool = thrd_ids_in_pool.back();
    thrd_ids_in_pool.pop_back();
    return thrd_id_from_pool;
  }

  int64_t GetThrdNum(std::pair<int32_t, int32_t> machine_task_type) {
    auto it = machine_task_type2thrd_num_.find(machine_task_type);
    CHECK(it != machine_task_type2thrd_num_.end());
    return it->second;
  }

  int32_t GetPreThrdId(int32_t machine_id, int32_t task_type);

  const JobDesc* desc_ = Global<JobDesc>::Get();
  const int64_t BASE_THRD_ID = desc_->resource().gpu_device_num() * 4
                               + desc_->resource().cpu_device_num() + 1;  // gpu cpu comm_net
  std::map<std::pair<int32_t, int32_t>, int32_t> machine_task_type2thrd_num_;

  HashMap<std::pair<int32_t, int32_t>, bool> machine_task_type2assigned_;

  HashMap<std::pair<int32_t, int32_t>, std::vector<int64_t>> machine_task_type2thrd_ids_;
  HashMap<std::pair<int32_t, int32_t>, std::vector<int64_t>> machine_task_type2thrd_id_pool_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
