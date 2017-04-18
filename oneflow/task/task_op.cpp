#include "task/task_op.h"
#include <vector>
#include <memory>
#include <glog/logging.h>
//#include "common/common.h"
//#include "context/one.h"
//#include "thread/base_thread.h"
//#include "task/device_context.h"
//#include "task/task.h"
//#include "task/task_item.h"
//#include "task/task_param.h"
//#include "task/task_consequence.h"
//#include "task/task_param_creator.h"
//#include "task/task_context.h"
//#include "dag/task_dag.h"
//
namespace oneflow {
//TaskOp::TaskOp(
//  Task<Dtype>* task,
//  std::shared_ptr<DeviceContext<Dtype>> device_context) :
//  task_(task),
//  device_context_(device_context), task_param_num_(10) {
//  // TODO(jiyuan): set the |task_param_num_|
//}
TaskOp::TaskOp(Task* task) : task_(task) {}

TaskOp::~TaskOp() { }

void TaskOp::Setup() {
  //task_param_creator_.reset(new TaskParamCreator<Dtype>(task_param_num_, task_));
  //InitContextParam();
}

//template <typename Dtype>
//void TaskOp<Dtype>::InitContextParam() {
//  if (task_->task_type() == TaskType::kCopyTask) {
//    context_param_.cuda_stream = device_context_->cuda_stream(task_->task_id());
//  } else if (task_->task_type() == TaskType::kComputeTask) {
//    context_param_.cublas_handle = device_context_->cublas_handle();
//    context_param_.cudnn_handle = device_context_->cudnn_handle();
//    context_param_.cuda_stream = device_context_->cuda_stream(task_->task_id());
//  }
//}

void TaskOp::Execute(TaskItem* task_item) {
  //// TODO(jiyuan): carefully handle with network task

  //// 1, Prepare execution environment based on task_item
  //auto& register_ids = task_item->register_ids();
  //auto& data_params = GetDataParams(register_ids);
  //auto& model_params = GetModelParams(register_ids);

  //// 2, Work on |task_item|, either Forward or Backward
  //if (task_->is_forward()) {
  //  Forward(data_params, model_params);
  //} else {
  //  Backward(data_params, model_params);
  //}

  //// 3, Register or do OnComplete routine
  //auto style = task_->task_consequence()->style();
  //switch (style) {
  //case ConsequenceStyle::kSynchronous:
  //  TaskConsequence<Dtype>::OnCompleteTaskItem(task_item);
  //  break;
  //case ConsequenceStyle::kAsynchronousCallback:
  //  // Add callback to the stream
  //  CUDA_CHECK(cudaStreamAddCallback(context_param_.cuda_stream,
  //    TaskConsequence<Dtype>::Callback,
  //    task_item, 0));
  //  break;
  //case ConsequenceStyle::kAsynchronousMessage:
  //  TaskConsequence<Dtype>::OnNetworkTaskItem(task_item);
  //  break;
  //default:
  //  LOG(FATAL) << "Unknown ConsequenceStyle";
  //  break;
  //}
}

//template <typename Dtype>
//const std::vector<DataParam<Dtype>*>&
//TaskOp<Dtype>::GetDataParams(
//const std::vector<int64_t>& register_ids) const {
//  auto task_param = task_param_creator_->GetTaskParam(register_ids);
//  return task_param->data_params();
//}
//
//template <typename Dtype>
//const std::vector<ModelParam<Dtype>*>&
//TaskOp<Dtype>::GetModelParams(
//const std::vector<int64_t>& register_ids) const {
//  auto task_param = task_param_creator_->GetTaskParam(register_ids);
//  return task_param->model_params();
//}
//
//template <typename Dtype>
//void TaskOp<Dtype>::Forward(
//  const std::vector<DataParam<Dtype>* >& data_params,
//  const std::vector<ModelParam<Dtype>* >& model_params) {
//  auto& ordered_layers = task_param_creator_->ordered_layers();
//  int32_t layer_num = ordered_layers.size();
//  for (int32_t i = 0; i < layer_num; ++i) {
//    ordered_layers[i]->Forward(context_param_, data_params[i], model_params[i]);
//  }
//}
//
//template <typename Dtype>
//void TaskOp<Dtype>::Backward(
//  const std::vector<DataParam<Dtype>* >& data_params,
//  const std::vector<ModelParam<Dtype>* >& model_params) {
//  auto& ordered_layers = task_param_creator_->ordered_layers();
//  int32_t layer_num = ordered_layers.size();
//  for (int32_t i = 0; i < layer_num; ++i) {
//    ordered_layers[i]->Backward(context_param_, data_params[i], model_params[i]);
//  }
//}

}
