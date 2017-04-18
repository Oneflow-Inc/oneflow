#ifndef _TASK_CONSEQUENCE_H_
#define _TASK_CONSEQUENCE_H_
#include <memory>
#include <vector>
#include "common/common.h"

namespace oneflow {
/*
TaskConsequence describes the consequence of task execution. This new class will
handle with the things like Callback.
*/
template <typename Dtype>
class Task;

class TaskItem;

enum class ConsequenceStyle {
  kSynchronous = 0,          // Used in kDataTask, kBoxingTask
  kAsynchronousCallback,     // Used in kCopyTask, kComputeTask
  kAsynchronousMessage       // Used in kNetTask
};

template <typename Dtype>
class TaskConsequence {
 public:
  explicit TaskConsequence(Task<Dtype>* task);
  ~TaskConsequence();

  ConsequenceStyle style() const { return style_; }
  static void OnCompleteTaskItem(TaskItem* task_item);
  static void CUDART_CB Callback(
    cudaStream_t stream, cudaError_t status, void* userData);
  static void OnNetworkTaskItem(TaskItem* task_item);

 private:
  Task<Dtype>* task_;
  ConsequenceStyle style_;

  TaskConsequence(const TaskConsequence& other) = delete;
  TaskConsequence& operator=(const TaskConsequence& other) = delete;
};
}  // namespace oneflow
#endif  // _TASK_CONSEQUENCE_H_
