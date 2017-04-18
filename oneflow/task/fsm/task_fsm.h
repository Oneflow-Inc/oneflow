#ifndef _TASK_FSM_H_
#define _TASK_FSM_H_
#include <glog/logging.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include "task/task.h"
#include "task/fsm/tetris.h"
#include "runtime/event_message.h"
namespace oneflow {
template <typename Dtype>
class Task;

class TaskItem;

template <typename Dtype>
class TaskFSM {
 public:
  explicit TaskFSM(Task<Dtype>* task) :task_(task),
    consumed_group_ids_(task->task_dag()->GetConsumedGroupIds()),
    produced_group_ids_(task->task_dag()->GetProducedGroupIds()) {}
  virtual ~TaskFSM() {}

  virtual void ProcessMessage(MsgPtr msg);
  virtual bool HasTaskItem() const;
  virtual TaskItem* GetTaskItem() = 0;

 protected:
  Task<Dtype>* task_;

  std::vector<int64_t> produced_group_ids_;
  std::vector<int64_t> consumed_group_ids_;

  Tetris<int64_t> tetris_;
  const int32_t source_col_id_{ -1 };
  std::unordered_map<int64_t, int32_t> group_id_to_col_id_;

  int32_t register_id_to_col_id(int64_t register_id);
};

template <typename Dtype>
void TaskFSM<Dtype>::ProcessMessage(MsgPtr msg) {
}

template <typename Dtype>
bool TaskFSM<Dtype>::HasTaskItem() const {
  return tetris_.Ready();
}

template <typename Dtype>
int32_t TaskFSM<Dtype>::register_id_to_col_id(int64_t register_id) {
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  auto group_id = id_map->group_id_from_register_id(register_id);
  return group_id_to_col_id_[group_id];
}

}  // namespace oneflow
#endif  // _TASK_FSM_H_
