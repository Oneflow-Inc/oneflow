#include "thread/base_thread.h"
#include "common/common.h"
#include "context/machine_descriptor.h"
#include "device/device_descriptor.h"
#include "task/device_manager.h"
#include "task/device_context.h"
#include "task/task.h"

namespace caffe {
template <typename Dtype>
BaseThread<Dtype>::BaseThread(MessageQueue message_queue) :
  thread_(nullptr), message_queue_(message_queue) {
  // device_context_(nullptr),
  // device_manager_(nullptr) {
}

template <typename Dtype>
BaseThread<Dtype>::~BaseThread() {
}

template <typename Dtype>
void BaseThread<Dtype>::Init() {
  //device_manager_.reset(
  //  new DeviceManager<Dtype>());
  //device_manager_->Setup();
  // Print for debug
  // device_manager_->PrintDeviceRegisters();
}

template <typename Dtype>
void BaseThread<Dtype>::Start() {
  thread_.reset(new std::thread(&BaseThread::ThreadMain, this));
  return;
}

template <typename Dtype>
void BaseThread<Dtype>::Join() {
  if (thread_) {
    thread_->join();
  }
}

template <typename Dtype>
void BaseThread<Dtype>::ThreadMain() {
  // TODO(jiyuan): how to gracefully exit?
  while (true) {
    MsgPtr msg;
    if (!message_queue_->Pop(msg)) {
      break;
    }
    int32_t to_task_id = msg->to_task_id();
    std::shared_ptr<Task<Dtype>> task = device_manager_->GetTask(to_task_id);
    task->ProcessMessage(msg);
  }
}
INSTANTIATE_CLASS(BaseThread);
}  // namespace caffe
