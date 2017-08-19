#include "oneflow/core/network/network_message_queue.h"

#include <glog/logging.h>
#include "oneflow/core/network/network.h"

namespace oneflow {

NetworkMessageQueue::NetworkMessageQueue() { network_ = GetRdmaInstance(); }

bool NetworkMessageQueue::TryPop(MsgPtr& msg) {
  if (!network_->Poll(&result_)) {
    // No event occurs at network
    return false;
  }
  if (result_.type == NetworkResultType::kSendOk) {
    // The network actor just ignore the kSendOk event
    return false;
  }
  msg.reset(new ActorMsg());
  switch (result_.type) {
    case NetworkResultType::kSendOk: return false;
    case NetworkResultType::kReceiveMsg: ProcessReceiveOK(msg); break;
    case NetworkResultType::kReadOk: ProcessReadOK(msg); break;
  }
  return true;
}

// TODO(shiyuan)
void NetworkMessageQueue::ProcessReceiveOK(MsgPtr& msg) {
  auto& net_msg = result_.net_msg;
  // There is only one expected type of message: MSG_TYPE_REQUEST_ACK
  CHECK(net_msg.type == NetworkMessageType::kRequestAck);
  *msg = net_msg.actor_msg;
  result_.callback(net_msg);
}

// TODO(shiyuan)
void NetworkMessageQueue::ProcessReadOK(MsgPtr& msg) {
  *msg = result_.net_msg.actor_msg;
  result_.callback(result_.net_msg);
}

}  // namespace oneflow
