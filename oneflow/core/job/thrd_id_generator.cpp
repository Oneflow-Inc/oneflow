#include "oneflow/core/job/thrd_id_generator.h"

namespace oneflow {

ThrdIdGenerator::ThrdIdGenerator(std::vector<std::pair<int64_t, TaskType>>& machine_task_type_vec,
                                 int64_t base_thrd_id)
    : base_thrd_id_(base_thrd_id) {
  HashMap<int64_t, std::set<TaskType>> machine2task_types;
  // machine_task_type = <machine_id, task_type>
  for (const auto machine_task_type : machine_task_type_vec) {
    if (EqualConf(machine_task_type.second, machine_task_type2thrd_num_[machine_task_type])) {
      continue;
    }
    machine_task_type2thrd_num_[machine_task_type]++;
    machine2task_types[machine_task_type.first].emplace(machine_task_type.second);
  }

  InitLowerboundOfTaskType(machine2task_types);
}

void ThrdIdGenerator::InitLowerboundOfTaskType(
    const HashMap<int64_t, std::set<TaskType>>& machine2task_types) {
  for (const auto& pair : machine2task_types) {
    int64_t machine_id = pair.first;
    auto& task_types = pair.second;

    int64_t lowerbound = base_thrd_id_;
    for (int64_t task_type : task_types) {
      auto machine_task_type = std::make_pair(machine_id, task_type);
      machine_task_type2lowerbound_[machine_task_type] = lowerbound;
      lowerbound += machine_task_type2thrd_num_[machine_task_type];
    }
  }
}

}  // namespace oneflow
