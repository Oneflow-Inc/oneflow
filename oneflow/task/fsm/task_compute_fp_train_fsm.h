#ifndef _TASK_COMPUTE_FP_TRAIN_FSM_H_
#define _TASK_COMPUTE_FP_TRAIN_FSM_H_
#include <glog/logging.h>
#include <vector>
#include <memory>
#include <cstdint>
#include "task/fsm/task_fsm.h"
#include "runtime/event_message.h"
namespace oneflow {
template <typename Dtype>
class Task;

class TaskItem;

template <typename Dtype>
class TaskComputeFPTrainFSM : public TaskFSM<Dtype> {
 public:
  explicit TaskComputeFPTrainFSM(Task<Dtype>* task);
  virtual ~TaskComputeFPTrainFSM();

  TaskItem* GetTaskItem() override;

 private:
  int32_t bp_compute_task_id_;
  std::vector<int32_t> consumer_ids_;

  TaskComputeFPTrainFSM(const TaskComputeFPTrainFSM& other) = delete;
  TaskComputeFPTrainFSM& operator=(const TaskComputeFPTrainFSM& other) = delete;
};
}  // namespace oneflow
#endif  // _TASK_COMPUTE_FP_TRAIN_FSM_H_
