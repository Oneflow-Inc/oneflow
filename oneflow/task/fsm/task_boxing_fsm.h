#ifndef _TASK_BOXING_FSM_H_
#define _TASK_BOXING_FSM_H_
#include <glog/logging.h>
#include <unordered_map>
#include <utility>
#include <memory>
#include <cstdint>
#include "task/fsm/task_fsm.h"
#include "runtime/event_message.h"
namespace oneflow {
template <typename Dtype>
class Task;

class TaskItem;

template <typename Dtype>
class TaskBoxingFSM : public TaskFSM<Dtype> {
 public:
  explicit TaskBoxingFSM(Task<Dtype>* task);
  virtual ~TaskBoxingFSM();

  TaskItem* GetTaskItem() override;

 private:
  std::unordered_map<int64_t, int32_t> group_id_to_consumer_id_;

  TaskBoxingFSM(const TaskBoxingFSM& other) = delete;
  TaskBoxingFSM& operator=(const TaskBoxingFSM& other) = delete;
};
}  // namespace oneflow
#endif  // _TASK_BOXING_FSM_H_
