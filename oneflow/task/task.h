#ifndef _TASK_H_
#define _TASK_H_
#include <glog/logging.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "common/task_type.h"
#include "thread/event_message.h"

/*
Task is a wrapper of some sub-classes, facilitating the task execution:

1, TaskDag describes a sub-graph consisting of data nodes and operator nodes.
TaskDag is the result of compile-time, forming an intermediate representation
between compile-time and run-time. Basically, all the required information about
a particular task can be looked-up from TaskDag (e.g., operators, memory
consumption, dependencies). However, it is still not that straightforward to
task execution. That is why we create the following objects from TaskDag to
facilitate the task execution.

2, TaskFSM maintains the finite state machine of a task, receiving messages,
updating internal states and triggering the execution of TaskOp.

3, TaskOp organizes the operators in TaskDag in a topologically sorted order.
In this way, while executing a task, we don't need to traverse the TaskDag
every time. TaskOp also does some preparations for task execution, such as
setting DataParam and ModelParam, retrieving appropriate resource (e.g., handle,
cuda stream) from TaskContext.

4, TaskContext manages all the allocated resources to this task, including memory
(e.g., data path and model path), cuda related handles, cuda streams.

5, TaskConsequence describes the consequence of an execution of the task, such
as sending a message to another particular task.
*/

namespace oneflow {
template <typename Dtype>
class TaskDag;

template <typename Dtype>
class TaskOp;

template <typename Dtype>
class TaskFSM;

template <typename Dtype>
class TaskContext;

template <typename Dtype>
class TaskConsequence;

template <typename Dtype>
class DeviceContext;

template <typename Dtype>
class Task {
 public:
  Task(std::shared_ptr<DeviceContext<Dtype>> device_context,
    std::shared_ptr<TaskDag<Dtype>> task_dag);
  ~Task();

  void Setup();
  void Release();
  void ProcessMessage(MsgPtr message);

  std::shared_ptr<TaskDag<Dtype>> task_dag() const;
  std::shared_ptr<TaskFSM<Dtype>> task_fsm() const;
  std::shared_ptr<TaskConsequence<Dtype>> task_consequence() const;
  std::shared_ptr<TaskOp<Dtype>> task_op() const;
  std::shared_ptr<TaskContext<Dtype>> task_context() const;

  bool is_net_receiver() const { return is_net_receiver_; }
  bool is_forward() const { return is_forward_; }
  int32_t task_id() const { return task_id_; }
  const std::string& task_name() const { return task_name_; }
  TaskType task_type() const { return task_type_; }
  bool is_h2d() const { CHECK(task_type_ == TaskType::kCopyTask); return is_h2d_; }

 private:
  bool is_forward_;
  int32_t task_id_;
  std::string task_name_;
  TaskType task_type_;
  bool is_h2d_;   // Valid only if task_type_ == TaskType::kCopyTask
  bool is_net_receiver_;   // Valid only if task_type_ == TaskType::kNetTask
  std::shared_ptr<DeviceContext<Dtype>> device_context_;

  std::shared_ptr<TaskDag<Dtype>> task_dag_;
  std::shared_ptr<TaskFSM<Dtype>> task_fsm_;
  std::shared_ptr<TaskConsequence<Dtype>> task_consequence_;
  std::shared_ptr<TaskOp<Dtype>> task_op_;
  std::shared_ptr<TaskContext<Dtype>> task_context_;

  Task(const Task& other) = delete;
  Task& operator=(const Task& other) = delete;
};
}  // namespace oneflow
#endif  // _TASK_H_
