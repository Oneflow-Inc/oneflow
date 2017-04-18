#ifndef _TASK_COMPUTE_MODEL_FSM_H_
#define _TASK_COMPUTE_MODEL_FSM_H_
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
class TaskComputeModelFSM : public TaskFSM<Dtype> {
 public:
  explicit TaskComputeModelFSM(Task<Dtype>* task);
  virtual ~TaskComputeModelFSM();

  void ProcessMessage(MsgPtr msg) override;
  bool HasTaskItem() const override;
  TaskItem* GetTaskItem() override;

 private:
  int64_t gradient_register_id_;  // we assume all diff arrive in data-id order
  std::queue<std::pair<int64_t, int64_t>> diff_ids_;  // diff id and version
  uint32_t added_diff_num_;
  uint32_t diff_num_;

  int16_t model_inited_;
  // 0: not init,
  // 1: newest_model_register_id_ is the init model,
  // 2: newest_model_register_id_ newer than the init model
  int64_t newest_model_register_id_;  // how to init with LoadPath?

  std::vector<int32_t> consumer_ids_;

  TaskComputeModelFSM(const TaskComputeModelFSM& other) = delete;
  TaskComputeModelFSM& operator=(const TaskComputeModelFSM& other) = delete;
};
}  // namespace oneflow
#endif  // _TASK_COMPUTE_MODEL_FSM_H_
