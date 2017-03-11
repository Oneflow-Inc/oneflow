#include "task/task_consequence.h"
#include "common/common.h"
#include "task/task_item.h"
#include "task/task.h"
#include "thread/event_message.h"
#include "context/one.h"
#include "thread/comm_bus.h"
#include "context/id_map.h"
#include "dag/task_dag.h"
#include "net/network.h"

namespace caffe {
template <typename Dtype>
TaskConsequence<Dtype>::TaskConsequence(
  Task<Dtype>* task) : task_(task) {
  switch (task_->task_type()) {
  case TaskType::kDataTask:
  case TaskType::kBoxingTask:
    style_ = ConsequenceStyle::kSynchronous;
    break;
  case TaskType::kCopyTask:
  case TaskType::kComputeTask:
    style_ = ConsequenceStyle::kAsynchronousCallback;
    break;
  case TaskType::kNetTask:
    style_ = ConsequenceStyle::kAsynchronousMessage;
    break;
  default:
    LOG(FATAL) << "Unknown TaskType";
    break;
  }
}

template <typename Dtype>
TaskConsequence<Dtype>::~TaskConsequence() {}

template <typename Dtype>
void CUDART_CB TaskConsequence<Dtype>::Callback(
  cudaStream_t stream, cudaError_t status, void* userData) {
  TaskItem* task_item = static_cast<TaskItem*>(userData);
  OnCompleteTaskItem(task_item);
}
template <typename Dtype>
void TaskConsequence<Dtype>::OnCompleteTaskItem(TaskItem* task_item) {
  auto& comm_bus = caffe::TheOne<Dtype>::comm_bus();
  auto msgs = task_item->msgs();
  for (auto msg : msgs) {
    comm_bus->SendMessage(msg);
  }
  //auto& id_map = caffe::TheOne<Dtype>::id_map();
  //auto from_task_id = item_ptr->task_id();
  //auto data_id = item_ptr->data_id();

  //// Send Ack message to source tasks
  //auto& src_register_ids = item_ptr->src_register_ids();
  //for (auto src_register_id : src_register_ids) {
  //  // TODO(jiyuan): sometime, the to_task_id is not got in this way
  //  int32_t to_task_id = id_map->task_id_from_register_id(src_register_id);
  //  MsgPtr message(new EventMessage(from_task_id,
  //    to_task_id, data_id, src_register_id));
  //  comm_bus->SendMessage(message);
  //}

  //// Send Ready messages to consumer tasks
  //auto& consumer_task_ids = item_ptr->consumer_task_ids();
  //auto dst_register_id = item_ptr->dst_register_id();
  //for (auto consumer_task_id : consumer_task_ids) {
  //  MsgPtr message(new EventMessage(from_task_id,
  //    consumer_task_id, data_id, dst_register_id));
  //  comm_bus->SendMessage(message);
  //}

  // Release the allocated TaskItem object
  delete task_item;
}

template <typename Dtype>
void TaskConsequence<Dtype>::OnNetworkTaskItem(TaskItem* task_item) {
  Network* network = GetNdspiRDMAInstance();
  auto& comm_bus = caffe::TheOne<Dtype>::comm_bus();
  auto msgs = task_item->msgs();
  for (auto& msg : msgs) {
    if (msg->message_type() == MessageType::kConsumed) {
      // Send the kConsumed event_message to remote peer
      comm_bus->SendMessage(msg);
    } else if (msg->message_type() == MessageType::kProduced) {
      // Register the kProduced event_message at Network object
      network->RegisterEventMessage(msg);
    }
  }
  delete task_item;
}
INSTANTIATE_CLASS(TaskConsequence);
}
