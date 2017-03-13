#ifndef _NET_THREAD_H_
#define _NET_THREAD_H_
#include <memory>
#include <thread>
#include "thread/base_thread.h"
#include "net/network_message_queue.h"

namespace caffe {

template <typename Dtype>
class NetThread : public BaseThread<Dtype> {
  public:
    NetThread(MessageQueue message_queue);
    virtual ~NetThread();

    NetThread(const NetThread& other) = delete;
    NetThread& operator=(const NetThread& other) = delete;

  private:
    std::shared_ptr<NetworkMessageQueue> net_message_queue_;
    void ThreadMain() override;
};
}  // namespace caffe
#endif  // _NET_THREAD_H_
