#include "task/task.h"
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "context/one.h"
#include "common/common.h"
#include "task/task_context.h"
#include "task/device_register.h"
#include "memory/memory_manager.h"
#include "task/task_item.h"
#include "task/task_op.h"
#include "task/task_consequence.h"
#include "task/fsm/task_fsm.h"
#include "context/id_map.h"
#include "context/config_parser.h"
#include "context/solver_descriptor.h"
#include "task/fsm/task_fsm_factory.h"
#include "task/device_context.h"

namespace oneflow {

template <typename Dtype>
Task<Dtype>::Task(std::shared_ptr<DeviceContext<Dtype>> device_context)
  : device_context_(device_context),
  is_forward_(true),
  task_id_(0)
  {}

template <typename Dtype>
Task<Dtype>::~Task() {}

template <typename Dtype>
void Task<Dtype>::Setup() {
  //FixMe xiaoshu
}

template <typename Dtype>
void Task<Dtype>::Release() {
  // task_context_->Release();
}

template <typename Dtype>
void Task<Dtype>::ProcessMessage(MsgPtr message) {
  task_fsm_->ProcessMessage(message);
  bool hasTaskItem = task_fsm_->HasTaskItem();
  if (hasTaskItem) {
    // |task_item| is allocated by |task_fsm_| and will be released by |task_op_|
    TaskItem *task_item = task_fsm_->GetTaskItem();
    task_op_->Execute(task_item);
  }
}


template <typename Dtype>
std::shared_ptr<TaskFSM<Dtype>> Task<Dtype>::task_fsm() const {
  return task_fsm_;
}

template <typename Dtype>
std::shared_ptr<TaskConsequence<Dtype>> Task<Dtype>::task_consequence() const {
  return task_consequence_;
}

template <typename Dtype>
std::shared_ptr<TaskOp<Dtype>> Task<Dtype>::task_op() const {
  return task_op_;
}

template <typename Dtype>
std::shared_ptr<TaskContext<Dtype>> Task<Dtype>::task_context() const {
  return task_context_;
}

INSTANTIATE_CLASS(Task);
}  // namespace oneflow
