
#include "thrd_id_generator.h"

namespace oneflow {

int64_t ThrdIdGenerator::GenerateThrdId(int32_t machine_id, int32_t task_type) {
  auto key = std::make_pair(machine_id, task_type);
  const int32_t thrd_num = GetThrdNum(key);
  const int32_t pre_thrd_id = GetPreThrdId(machine_id, task_type);

  int64_t thrd_id_from_pool = GetThrdIdFromPool(key);
  int64_t thrd_id = pre_thrd_id + thrd_id_from_pool;

  // save the thread id
  auto it = machine_task_type2thrd_ids_.find(key);
  if (it == machine_task_type2thrd_ids_.end()) {
    std::vector<int64_t> thrd_ids{thrd_id};
    machine_task_type2thrd_ids_.emplace(key, std::move(thrd_ids));
  } else {
    auto& thrd_ids = it->second;
    if (thrd_ids.size() < thrd_num) thrd_ids.push_back(thrd_id);
  }

  // set assigned flag
  machine_task_type2assigned_[key] = true;

  return thrd_id;
}

int32_t ThrdIdGenerator::GetPreThrdId(int32_t machine_id, int32_t task_type) {
  auto cur_key = std::make_pair(machine_id, task_type);
  int32_t total = 0;
  for (auto pair : machine_task_type2assigned_) {
    if (pair.first == cur_key) { continue; }

    if (pair.second) {
      const int32_t thrd_num = GetThrdNum(pair.first);
      total += thrd_num;
    }
  }

  total += BASE_THRD_ID;
  return total;
}
}  // namespace oneflow
