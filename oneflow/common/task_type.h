#ifndef _DAG_TASK_TYPE_H_
#define _DAG_TASK_TYPE_H_
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
#endif  // _DAG_TASK_TYPE_H_
