#include "oneflow/core/job/thrd_id_generator.h"

namespace oneflow {

ThrdIdGenerator::ThrdIdGenerator(std::vector<std::pair<int64_t, TaskType>>& machine_task_type_vec,
                                 int64_t base_thrd_id)
    : base_thrd_id_(base_thrd_id) {
  HashMap<int64_t, std::set<TaskType>> machine2task_types;
  for (const auto pair : machine_task_type_vec) {
    int64_t machine_id = pair.first;
    auto key = std::make_pair(machine_id, pair.second);
    if (EqualConf(pair.second, machine_task_type2thrd_num_[key])) { continue; }
    machine_task_type2thrd_num_[key]++;

    machine2task_types[machine_id].emplace(pair.second);
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
      auto key = std::make_pair(machine_id, task_type);
      machine_task_type2lowerbound_[key] = lowerbound;
      lowerbound += machine_task_type2thrd_num_[key];
    }
  }
}

}  // namespace oneflow
