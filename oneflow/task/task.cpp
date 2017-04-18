#include "task/task.h"
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
//#include "dag/dag.h"
//#include "dag/dag_node.h"
//#include "dag/task_dag.h"
//#include "dag/node_meta.h"
//#include "context/one.h"
//#include "common/common.h"
//#include "task/task_context.h"
//#include "task/device_register.h"
//#include "memory/memory_manager.h"
//#include "task/task_item.h"
//#include "task/task_op.h"
//#include "task/task_consequence.h"
//#include "task/fsm/task_fsm.h"
//#include "context/id_map.h"
//#include "context/config_parser.h"
//#include "context/solver_descriptor.h"
//#include "task/fsm/task_fsm_factory.h"
//#include "task/device_context.h"
//
namespace oneflow {

//template <typename Dtype>
//Task<Dtype>::Task(std::shared_ptr<DeviceContext<Dtype>> device_context,
//  std::shared_ptr<TaskDag<Dtype>> task_dag)
//  : device_context_(device_context),
//  task_dag_(task_dag),
//  is_forward_(task_dag->is_forward()),
//  task_id_(task_dag->task_id()),
//  task_name_(task_dag->actor_name()),
//  task_type_(task_dag->task_type()) {
//}

Task::Task() {}

// template <typename Dtype>
Task::~Task() {}

// template <typename Dtype>
void Task::Setup() {
  //LOG(INFO) << "Build task: " << task_name_ << std::endl;
  //if (task_type_ == TaskType::kCopyTask) {
  //  is_h2d_ = task_dag_->is_h2d();
  //}
  //if (task_type_ == TaskType::kNetTask) {
  //  is_net_receiver_ = task_dag_->is_net_receiver();
  //}
  //task_context_.reset(new TaskContext<Dtype>(this));
  //task_context_->Setup();

  //task_fsm_ = TaskFSMFactory<Dtype>::CreateFSM(this);
  //task_consequence_.reset(new TaskConsequence<Dtype>(this));

  //// Register to |device_context_| before creating |task_op_|, since |task_op_|
  //// needs get some resources from |device_context_|
  //device_context_->RegisterTask(this);

  //task_op_.reset(new TaskOp<Dtype>(this, device_context_));
  //task_op_->Setup();
}

void Task::Release() {
  // task_context_->Release();
}

void Task::ProcessMessage(MsgPtr message) {
  //task_fsm_->ProcessMessage(message);
  //bool hasTaskItem = task_fsm_->HasTaskItem();
  //if (hasTaskItem) {
  //  // |task_item| is allocated by |task_fsm_| and will be released by |task_op_|
  //  TaskItem *task_item = task_fsm_->GetTaskItem();
  //  task_op_->Execute(task_item);
  //}
}

//template <typename Dtype>
//std::shared_ptr<TaskDag<Dtype>> Task<Dtype>::task_dag() const {
//  return task_dag_;
//}
//
//template <typename Dtype>
//std::shared_ptr<TaskFSM<Dtype>> Task<Dtype>::task_fsm() const {
//  return task_fsm_;
//}
//
//template <typename Dtype>
//std::shared_ptr<TaskConsequence<Dtype>> Task<Dtype>::task_consequence() const {
//  return task_consequence_;
//}
//
//template <typename Dtype>
//std::shared_ptr<TaskOp<Dtype>> Task<Dtype>::task_op() const {
//  return task_op_;
//}
//
//template <typename Dtype>
//std::shared_ptr<TaskContext<Dtype>> Task<Dtype>::task_context() const {
//  return task_context_;
//}

}  // namespace oneflow
