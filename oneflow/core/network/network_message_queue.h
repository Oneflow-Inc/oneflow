#ifndef ONEFLOW_CORE_NETWORK_NETWORK_MESSAGE_QUEUE_H_
#define ONEFLOW_CORE_NETWORK_NETWORK_MESSAGE_QUEUE_H_
#include <memory>
#include "oneflow/core/network/network.h"
#include "oneflow/core/network/network_message.h"

namespace oneflow {

class NetworkMessageQueue {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NetworkMessageQueue);
  NetworkMessageQueue();
  ~NetworkMessageQueue() = default;

  bool TryPop(MsgPtr& msg);

 private:
  Network* network_;
  NetworkResult result_;

  void ProcessReceiveOK(MsgPtr& msg);
  void ProcessReadOK(MsgPtr& msg);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_NETWORK_MESSAGE_QUEUE_H_
