#include "task/fsm/task_fsm_factory.h"
#include "path/base_path.h"
#include "dag/task_dag.h"

namespace caffe {
template <typename Dtype>
std::shared_ptr<TaskFSM<Dtype>> TaskFSMFactory<Dtype>::CreateFSM(
  Task<Dtype>* task) {
  switch (task->task_type()) {
  case TaskType::kDataTask:
    return std::make_shared<TaskDataFSM<Dtype>>(task);
  case TaskType::kCopyTask:
  case TaskType::kNetTask:
    return std::make_shared<TaskCommonFSM<Dtype>>(task);
  case TaskType::kBoxingTask:
    return std::make_shared<TaskBoxingFSM<Dtype>>(task);
  case TaskType::kComputeTask:
  {
    switch (task->task_dag()->path_type()) {
    case PathType::kDataPath:
      if (task->task_dag()->is_forward()) {
        // TODO(v-kayin): phase TEST or TRAIN?
        bool is_train = true;
        if (is_train) {
          return std::make_shared<TaskComputeFPTrainFSM<Dtype>>(task);
        } else {
          return std::make_shared<TaskComputeFPTestFSM<Dtype>>(task);
        }
      } else {
        return std::make_shared<TaskComputeBPFSM<Dtype>>(task);
      }
    case PathType::kModelUpdatePath:
      return std::make_shared<TaskComputeModelFSM<Dtype>>(task);
    case PathType::kModelLoadPath:
    case PathType::kModelStorePath:
      return std::make_shared<TaskCommonFSM<Dtype>>(task);
    default:
      CHECK(false);
      break;
    }
  }
  default:
    CHECK(false);
    break;
  }
}

INSTANTIATE_CLASS(TaskFSMFactory);
}  // namespace caffe
