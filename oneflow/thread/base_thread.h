#ifndef _BASE_THREAD_H_
#define _BASE_THREAD_H_
#include <memory>
#include <thread>
#include "thread/event_message.h"
#include "thread/mt_queue.h"
#include "thread/comm_bus.h"

namespace caffe {
class DeviceDescriptor;

template <typename Dtype>
class DeviceManager;

template <typename Dtype>
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
    std::shared_ptr<DeviceManager<Dtype>> device_manager_;

    virtual void ThreadMain();
};
}  // namespace caffe
#endif  // _BASE_THREAD_H_
