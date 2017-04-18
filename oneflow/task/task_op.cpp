#include "task/task_op.h"
#include <vector>
#include <memory>
#include <glog/logging.h>
#include "common/common.h"
#include "context/one.h"
#include "runtime/base_thread.h"
#include "task/device_context.h"
#include "task/task.h"
#include "task/task_item.h"
#include "task/task_param.h"
#include "task/task_consequence.h"
#include "task/task_param_creator.h"
#include "task/task_context.h"

namespace oneflow {
template <typename Dtype>
TaskOp<Dtype>::TaskOp(
  Task<Dtype>* task,
  std::shared_ptr<DeviceContext<Dtype>> device_context) :
  task_(task),
  device_context_(device_context), task_param_num_(10) {
  // TODO(jiyuan): set the |task_param_num_|
}

template <typename Dtype>
TaskOp<Dtype>::~TaskOp() { }

template <typename Dtype>
void TaskOp<Dtype>::Setup() {
  task_param_creator_.reset(new TaskParamCreator<Dtype>(task_param_num_, task_));
  InitContextParam();
}

template <typename Dtype>
void TaskOp<Dtype>::InitContextParam() {
  if (task_->task_type() == TaskType::kCopyTask) {
  } else if (task_->task_type() == TaskType::kComputeTask) {
  }
}

template <typename Dtype>
void TaskOp<Dtype>::Execute(TaskItem* task_item) {
  //FixMe xiaoshu
}


template <typename Dtype>
void TaskOp<Dtype>::Forward() {
  //FixMe xiaoshu
}

template <typename Dtype>
void TaskOp<Dtype>::Backward() {
  //FixMe
}

INSTANTIATE_CLASS(TaskOp);
}
