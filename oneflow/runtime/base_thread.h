#ifndef ONEFLOW_RUNTIME_BASE_THREAD_H_
#define ONEFLOW_RUNTIME_BASE_THREAD_H_
#include <memory>
#include <thread>
#include "runtime/event_message.h"
#include "runtime/mt_queue.h"
#include "runtime/comm_bus.h"

namespace oneflow {
// class DeviceDescriptor;

// template <typename Dtype>
// class DeviceManager;

class BaseThread {
  public:
    BaseThread(MessageQueue message_queue);
    virtual ~BaseThread();

    void Init();
    void Start();
    void Join();

  protected:
    MessageQueue message_queue_;
    int32_t thread_id_;
    std::unique_ptr<std::thread> thread_;
    // std::shared_ptr<DeviceManager<Dtype>> device_manager_;

    virtual void ThreadMain();
};
}  // namespace oneflow
#endif  // ONEFLOW_RUNTIME_BASE_THREAD_H_
