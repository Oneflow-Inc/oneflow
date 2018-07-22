
#include "thrd_id_distributor.h"

namespace oneflow {

ThrdIdDistributor::ThrdIdDistributor() {
  task_type2thrd_num_.emplace(TaskType::kCopyCommNet, 1);
  task_type2thrd_num_.emplace(TaskType::kMdSave, Global<JobDesc>::Get()->MdSaveWorkerNum());

  task_type2assigned_ = {{TaskType::kMdSave, false},
                         {TaskType::kLossPrint, false},
                         {TaskType::kRecordLoad, false},
                         {TaskType::kCopyCommNet, false}};
}

int64_t ThrdIdDistributor::GenerateThrdId(TaskType task_type, int32_t offset) {
  auto it = task_type2assigned_.find(task_type);
  assert(it != task_type2assigned_.end());

  int32_t pre_thrd_id = GetPreThrdId(task_type);

  it->second = true;

  const int32_t thrd_num = task_type2thrd_num_.at(task_type);
  int64_t thrd_id = pre_thrd_id + offset % thrd_num;
  auto& thrd_ids = task_type2thrd_ids_[task_type];
  if (thrd_ids.size() < thrd_num) thrd_ids.push_back(thrd_id);

  return thrd_id;
}

int32_t ThrdIdDistributor::GetPreThrdId(TaskType task_type) {
  int32_t total = 0;
  for (auto pair : task_type2assigned_) {
    if (pair.first == task_type) { continue; }

    if (pair.second) {
      auto it = task_type2thrd_num_.find(pair.first);
      if (it != task_type2thrd_num_.end()) total += it->second;
    }
  }

  total += BASE_THRD_ID;
  return total;
}

int64_t ThrdIdDistributor::GetThrdId(TaskType task_type, int32_t index) const {
  auto it = task_type2thrd_ids_.find(task_type);
  assert(it != task_type2thrd_ids_.end());
  auto& thrd_ids = it->second;
  assert(index < thrd_ids.size());
  return thrd_ids[index];
}
}  // namespace oneflow
