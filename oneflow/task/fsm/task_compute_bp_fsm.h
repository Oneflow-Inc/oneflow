#ifndef _TASK_COMPUTE_BP_FSM_H_
#define _TASK_COMPUTE_BP_FSM_H_
#include <glog/logging.h>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include "task/fsm/task_fsm.h"
#include "runtime/event_message.h"
namespace oneflow {
template <typename Dtype>
class Task;

class TaskItem;

template <typename Dtype>
class TaskComputeBPFSM : public TaskFSM<Dtype> {
 public:
  explicit TaskComputeBPFSM(Task<Dtype>* task);
  virtual ~TaskComputeBPFSM();

  TaskItem* GetTaskItem() override;

 private:
  int32_t fp_compute_task_id_;
  std::unordered_map<int64_t, int32_t> group_id_to_consumer_id_;

  TaskComputeBPFSM(const TaskComputeBPFSM& other) = delete;
  TaskComputeBPFSM& operator=(const TaskComputeBPFSM& other) = delete;
};
}  // namespace oneflow
#endif  // _TASK_COMPUTE_BP_FSM_H_
