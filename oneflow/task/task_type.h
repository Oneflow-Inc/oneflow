#ifndef ONEFLOW_TASK_TASK_TYPE_H_
#define ONEFLOW_TASK_TASK_TYPE_H_
namespace oneflow {
enum class TaskType {
  kUnknownTask = 0,
  kDataTask,
  kBoxingTask,
  kCopyTask,
  kNetTask,
  kComputeTask
};
}  // namespace oneflow
#endif  // ONEFLOW_TASK_TASK_TYPE_H_