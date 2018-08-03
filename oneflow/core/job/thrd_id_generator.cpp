#include "oneflow/core/job/thrd_id_generator.h"

namespace oneflow {

ThrdIdGenerator::ThrdIdGenerator(std::vector<std::pair<int64_t, TaskType>>& machine_task_type_vec,
                                 int64_t base_thrd_id)
    : base_thrd_id_(base_thrd_id) {
  HashMap<int64_t, std::vector<TaskType>> machine2task_type_seq;
  for (const auto pair : machine_task_type_vec) {
    int64_t machine_id = pair.first;
    auto key = std::make_pair(machine_id, pair.second);
    if (EqualConf(pair.second, machine_task_type2thrd_num_[key])) continue;
    machine_task_type2thrd_num_[key]++;

    machine2task_type_seq[machine_id].push_back(pair.second);
  }

  for (auto& pair : machine2task_type_seq) Unique(pair.second);
  InitLowerboundOfTaskType(machine2task_type_seq);
}

void ThrdIdGenerator::InitLowerboundOfTaskType(
    const HashMap<int64_t, std::vector<TaskType>>& machine2task_type_seq) {
  for (const auto& pair : machine2task_type_seq) {
    int64_t machine_id = pair.first;
    auto& task_type_seq = pair.second;
    int64_t lowerbound = base_thrd_id_;
    for (int i = 0; i < task_type_seq.size(); ++i) {
      int64_t task_type = task_type_seq[i];
      auto key = std::make_pair(machine_id, task_type);
      if (i == 0) {
        machine_task_type2lowerbound_[key] = lowerbound;
        continue;
      }

      auto last_key = std::make_pair(machine_id, task_type_seq[i - 1]);
      lowerbound += machine_task_type2thrd_num_[last_key];
      machine_task_type2lowerbound_[key] = lowerbound;
    }
  }
}

}  // namespace oneflow
