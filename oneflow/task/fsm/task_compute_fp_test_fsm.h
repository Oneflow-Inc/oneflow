#ifndef _TASK_COMPUTE_FP_TEST_FSM_H_
#define _TASK_COMPUTE_FP_TEST_FSM_H_
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
class TaskComputeFPTestFSM : public TaskFSM<Dtype> {
 public:
  explicit TaskComputeFPTestFSM(Task<Dtype>* task);
  virtual ~TaskComputeFPTestFSM();

  void ProcessMessage(MsgPtr msg) override;
  bool HasTaskItem() const override;
  TaskItem* GetTaskItem() override;

 private:
  int32_t consumer_id_;
  int64_t model_register_id_{ -1 };

  TaskComputeFPTestFSM(const TaskComputeFPTestFSM& other) = delete;
  TaskComputeFPTestFSM& operator=(const TaskComputeFPTestFSM& other) = delete;
};
}  // namespace oneflow
#endif  // _TASK_COMPUTE_FP_TEST_FSM_H_
