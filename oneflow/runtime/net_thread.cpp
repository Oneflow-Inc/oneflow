#include "runtime/net_thread.h"
// #include "common/common.h"
// #include "context/one.h"
#include "context/id_map.h"
#include "runtime/comm_bus.h"
// #include "task/device_manager.h"
// #include "task/task.h"

namespace oneflow {

NetThread::NetThread(MessageQueue message_queue) :
  BaseThread(message_queue) {
}
NetThread::~NetThread() {
}

void NetThread::ThreadMain() {
  MsgPtr msg;
  // auto& comm_bus = caffe::TheOne<Dtype>::comm_bus();
  // auto& id_map = caffe::TheOne<Dtype>::id_map();
  while (true) {
    if (message_queue_->TryPop(msg)) {
      int32_t to_task_id = msg->to_task_id();
      // std::shared_ptr<Task<Dtype>> task = device_manager_->GetTask(to_task_id);
      // task->ProcessMessage(msg);
    }
    if (net_message_queue_->TryPop(msg)) {
      std::shared_ptr<IDMap> id_map(new IDMap());  // FIXME(jiyuan)
      int32_t to_task_id = msg->to_task_id();
      int32_t to_thread_id = id_map->thread_id_from_task_id(to_task_id);
      if (to_thread_id != thread_id_) {
        // A READ action needs to trigger two messages: 1, a kConsumed
        // event_message to the producer (sender) at remote peer; 2, a kProduced
        // event_message to the consumer at the same machine. However, we can
        // not send the message from net_thread to the thread on which the
        // consumer resides immediately after we issue the READ verb. We must
        // wait until the READ completion event occurs. The READ completion
        // event can be polled from Network object. We let the completion event
        // carry the second event_message which should be sent to the consumer.
        // When the READ completion event occurs, the carried event_message will
        // be put in the |net_message_queue_|. Once we observe that the receiver
        // of the event_message resides on another thread, we know it must be
        // an instance of the second event_message, hence we forward it to the
        // real receiver.

        // comm_bus->SendMessage(msg);
      } else {
        // std::shared_ptr<Task<Dtype>> task = device_manager_->GetTask(to_task_id);
        // task->ProcessMessage(msg);
      }
    }
  }
}
}  // namespace oneflow