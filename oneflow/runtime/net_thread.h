#ifndef ONEFLOW_RUNTIME_NET_THREAD_H_
#define ONEFLOW_RUNTIME_NET_THREAD_H_
#include <memory>
#include <thread>
#include "runtime/base_thread.h"
#include "network/network_message_queue.h"

namespace oneflow {
template <typename Dtype>
class NetThread : public BaseThread<Dtype> {
  public:
    NetThread(MessageQueue message_queue);
    virtual ~NetThread();

    NetThread(const NetThread& other) = delete;
    NetThread& operator=(const NetThread& other) = delete;

  private:
    MessageQueue message_queue_;
    std::shared_ptr<NetworkMessageQueue> net_message_queue_;
    int32_t thread_id_;
    void ThreadMain() override;
};
}  // namespace oneflow
#endif  // ONEFLOW_RUNTIME_NET_THREAD_H_
