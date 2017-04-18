#ifndef ONEFLOW_RUNTIME_NET_THREAD_H_
#define ONEFLOW_RUNTIME_NET_THREAD_H_
#include <memory>
#include <thread>
#include "runtime/base_thread.h"
#include "network/network_message_queue.h"

namespace oneflow {

class NetThread : public BaseThread {
  public:
    NetThread(MessageQueue message_queue);
    virtual ~NetThread();

    NetThread(const NetThread& other) = delete;
    NetThread& operator=(const NetThread& other) = delete;

  private:
    std::shared_ptr<NetworkMessageQueue> net_message_queue_;
    void ThreadMain() override;
};
}  // namespace oneflow
#endif  // ONEFLOW_RUNTIME_NET_THREAD_H_
