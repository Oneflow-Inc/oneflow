#include "oneflow/core/job/thrd_id_generator.h"

namespace oneflow {

int64_t ThrdIdGenerator::GenerateThrdId(int64_t machine_id, int64_t task_type) {
  auto key = std::make_pair(machine_id, task_type);
  const int64_t pre_thrd_id = GetPreThrdId(machine_id, task_type);
  int64_t thrd_id = pre_thrd_id + GetThrdIdFromPool(key) + 1;

  return thrd_id;
}

int64_t ThrdIdGenerator::GetPreThrdId(int64_t machine_id, int64_t task_type) {
  if (machine_task_type2lowerbound_[machine_id].empty()) {
    machine_task_type2lowerbound_[machine_id][task_type] = base_thrd_id_;
    return base_thrd_id_;
  }

  if (machine_task_type2lowerbound_[machine_id][task_type] == 0) {
    int64_t latest_thrd_id = 0;
    int64_t distance = 0;
    for (auto pair : machine_task_type2lowerbound_[machine_id]) {
      if (pair.first == task_type) { continue; }

      if (latest_thrd_id < pair.second) {
        latest_thrd_id = pair.second;
        distance = machine_task_type2thrd_num_[std::make_pair(machine_id, pair.first)];
      }
    }

    int64_t pre_thrd_id = latest_thrd_id + distance;
    machine_task_type2lowerbound_[machine_id][task_type] = pre_thrd_id;
    return pre_thrd_id;
  } else {
    return machine_task_type2lowerbound_[machine_id][task_type];
  }
}
}  // namespace oneflow
