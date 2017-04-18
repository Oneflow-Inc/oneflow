#include "runtime/base_thread.h"
// #include "common/common.h"
// #include "context/machine_descriptor.h"
// #include "device/device_descriptor.h"
#include "task/device_manager.h"
// #include "task/device_context.h"
#include "task/task.h"

namespace oneflow {
BaseThread::BaseThread(MessageQueue message_queue) :
  thread_(nullptr), message_queue_(message_queue) {
  // device_context_(nullptr),
  // device_manager_(nullptr) {
}

BaseThread::~BaseThread() {
}

void BaseThread::Init() {
  //device_manager_.reset(
  //  new DeviceManager<Dtype>());
  //device_manager_->Setup();
  // Print for debug
  // device_manager_->PrintDeviceRegisters();
}

void BaseThread::Start() {
  thread_.reset(new std::thread(&BaseThread::ThreadMain, this));
  return;
}

void BaseThread::Join() {
  if (thread_) {
    thread_->join();
  }
}

void BaseThread::ThreadMain() {
  // TODO(jiyuan): how to gracefully exit?
  while (true) {
    MsgPtr msg;
    if (!message_queue_->Pop(msg)) {
      break;
    }
    int32_t to_task_id = msg->to_task_id();
    std::shared_ptr<Task> task = device_manager_->GetTask(to_task_id);
    task->ProcessMessage(msg);
  }
}
}  // namespace oneflow
