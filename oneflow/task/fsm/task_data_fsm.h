#ifndef _TASK_DATA_FSM_H_
#define _TASK_DATA_FSM_H_
#include <glog/logging.h>
#include <memory>
#include <cstdint>
#include "task/fsm/task_fsm.h"
#include "runtime/event_message.h"
namespace oneflow {
template <typename Dtype>
class Task;

class TaskItem;

template <typename Dtype>
class TaskDataFSM : public TaskFSM<Dtype> {
 public:
  explicit TaskDataFSM(Task<Dtype>* task);
  virtual ~TaskDataFSM();

  TaskItem* GetTaskItem() override;

 private:
  int32_t consumer_id_;
  // for data task, count piece_id and get new data id
  int64_t piece_id_;
  int32_t device_id_;
  int64_t new_data_id();

  TaskDataFSM(const TaskDataFSM& other) = delete;
  TaskDataFSM& operator=(const TaskDataFSM& other) = delete;
};
}  // namespace oneflow
#endif  // _TASK_DATA_FSM_H_
