#ifndef ONEFLOW_TASK_TASK_OP_H_
#define ONEFLOW_TASK_TASK_OP_H_
#include <memory>
#include <vector>
// #include "device/device_alternate.h"
#include "runtime/event_message.h"
// #include "layers/base_layer.h"
#include "task/task_type.h"

namespace oneflow {
class Task;
// class TaskContext;

// class DeviceContext;

class TaskItem;

class TaskParamCreator;

class TaskOp {
 public:
   // TaskOp(Task<Dtype>* task, std::shared_ptr<DeviceContext<Dtype>> device_context);
   TaskOp(Task* task);
  ~TaskOp();

  void Setup();
  void Execute(TaskItem* task_item);

 private:
  Task* task_;
  // ContextParam context_param_;
  // std::shared_ptr<DeviceContext<Dtype>> device_context_;

  // Indicates the size of TaskParam cache
  //int32_t task_param_num_;
  //std::shared_ptr<TaskParamCreator<Dtype>> task_param_creator_;

  //void Forward(const std::vector<DataParam<Dtype>*>& data_params,
  //  const std::vector<ModelParam<Dtype>*>& model_params);

  //void Backward(const std::vector<DataParam<Dtype>*>& data_params,
  //  const std::vector<ModelParam<Dtype>*>& model_params);

  //const std::vector<DataParam<Dtype>*>& GetDataParams(
  //  const std::vector<int64_t>& register_ids) const;
  //const std::vector<ModelParam<Dtype>*>& GetModelParams(
  //  const std::vector<int64_t>& register_ids) const;

  //void InitContextParam();

  TaskOp(const TaskOp& other) = delete;
  TaskOp& operator=(const TaskOp& other) = delete;
};

}  // namespace oneflow
#endif  // ONEFLOW_TASK_TASK_OP_H_
