#ifndef ONEFLOW_CORE_NETWORK_NETWORK_MESSAGE_QUEUE_H_
#define ONEFLOW_CORE_NETWORK_NETWORK_MESSAGE_QUEUE_H_
#include <memory>
#include "oneflow/core/network/network.h"
#include "oneflow/core/network/network_message.h"

// NetworkMessageQueue provides an friendly interface for NetThread which could
// polls and uses the network message just like it processes the local message.
// Internally, it uses Network object's Poll interface to get network message
// and transform the network event to EventMessage.
namespace oneflow {

class NetworkMessageQueue {
 public:
  NetworkMessageQueue();
  ~NetworkMessageQueue() = default;

  bool TryPop(MsgPtr& msg);

 private:
  Network* network_;
  NetworkResult result_;

  void ProcessReceiveOK(MsgPtr& msg);
  void ProcessReadOK(MsgPtr& msg);

  NetworkMessageQueue(const NetworkMessageQueue& other) = delete;
  NetworkMessageQueue& operator=(const NetworkMessageQueue& other) = delete;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_NETWORK_MESSAGE_QUEUE_H_
