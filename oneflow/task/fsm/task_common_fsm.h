#ifndef _TASK_COMMON_FSM_H_
#define _TASK_COMMON_FSM_H_
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
class TaskCommonFSM : public TaskFSM<Dtype> {
 public:
  explicit TaskCommonFSM(Task<Dtype>* task);
  virtual ~TaskCommonFSM();

  TaskItem* GetTaskItem() override;

 private:
  int32_t consumer_id_;

  TaskCommonFSM(const TaskCommonFSM& other) = delete;
  TaskCommonFSM& operator=(const TaskCommonFSM& other) = delete;
};
}  // namespace oneflow
#endif  // _TASK_COMMON_FSM_H_
