#ifndef ONEFLOW_TASK_TASK_FSM_H_
#define ONEFLOW_TASK_TASK_FSM_H_
#include <glog/logging.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include "context/id_map.h"
#include "task/task.h"
#include "task/tetris.h"
#include "runtime/event_message.h"

namespace oneflow {
class Task;
class TaskItem;

class TaskFSM {
 public:
  explicit TaskFSM(Task* task)
    // : task_(task),
    // consumed_group_ids_(task->task_dag()->GetConsumedGroupIds()),
    // produced_group_ids_(task->task_dag()->GetProducedGroupIds())
  {}
  virtual ~TaskFSM() {}

  virtual void ProcessMessage(MsgPtr msg);
  virtual bool HasTaskItem() const;
  virtual TaskItem* GetTaskItem() = 0;

 protected:
  Task* task_;

  std::vector<int64_t> produced_group_ids_;
  std::vector<int64_t> consumed_group_ids_;

  Tetris tetris_;
  const int32_t source_col_id_{ -1 };
  std::unordered_map<int64_t, int32_t> group_id_to_col_id_;

  int32_t register_id_to_col_id(int64_t register_id);
};

void TaskFSM::ProcessMessage(MsgPtr msg) {
  auto col_id = register_id_to_col_id(msg->register_id());
  // auto& id_map = caffe::TheOne<Dtype>::id_map();  // FIXME(jiyua)
  std::shared_ptr<IDMap> id_map(new IDMap());
  auto piece_id = id_map->piece_id_from_data_id(msg->data_id());
  tetris_.Push(col_id, msg->register_id(), piece_id);
}

bool TaskFSM::HasTaskItem() const {
  return tetris_.Ready();
}

int32_t TaskFSM::register_id_to_col_id(int64_t register_id) {
  // auto& id_map = caffe::TheOne<Dtype>::id_map();  // FIXME(jiyuan)
  std::shared_ptr<IDMap> id_map(new IDMap());
  auto group_id = id_map->group_id_from_register_id(register_id);
  return group_id_to_col_id_[group_id];
}

}  // namespace oneflow
#endif  // ONEFLOW_TASK_TASK_FSM_H_
