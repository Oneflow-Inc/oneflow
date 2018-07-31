#include "oneflow/core/job/thrd_id_generator.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

int64_t ThrdIdGenerator::GenerateThrdId(int64_t machine_id, int64_t task_type) {
  auto key = std::make_pair(machine_id, task_type);
  const int64_t pre_thrd_id = GetPreThrdId(machine_id, task_type);
  int64_t thrd_id = pre_thrd_id + GetThrdIdFromPool(key);

  // set assigned flag
  machine_task_type2assigned_[key] = true;

  return thrd_id;
}

int32_t ThrdIdGenerator::GetPreThrdId(int64_t machine_id, int64_t task_type) {
  auto cur_key = std::make_pair(machine_id, task_type);
  int32_t total = 0;
  for (auto pair : machine_task_type2assigned_) {
    if (pair.first == cur_key) { continue; }

    if (pair.second) {
      const int32_t thrd_num = GetThrdNum(pair.first);
      total += thrd_num;
    }
  }

  const int64_t BASE_THRD_ID = Global<IDMgr>::Get()->CommNetThrdId();  // gpu cpu comm_net
  total += BASE_THRD_ID;
  return total;
}
}  // namespace oneflow
