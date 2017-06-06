#include "network/network_message_queue.h"

// #include "common/common.h"
#include <glog/logging.h>
#include "network/network.h"
//#include "runtime/event_message.h"

namespace oneflow {

NetworkMessageQueue::NetworkMessageQueue() {
  network_ = GetRdmaInstance();
}

NetworkMessageQueue::~NetworkMessageQueue() {
}

bool NetworkMessageQueue::TryPop(MsgPtr& msg) {
  if (!network_->Poll(&result_)) {
    // No event occurs at network
    return false;
  }
  if (result_.type == NetworkResultType::NET_SEND_OK) {
    // The network actor just ignore the NET_SENT_OK event
    return false;
  }
  msg.reset(new ActorMsg());
  switch (result_.type) {
  case NetworkResultType::NET_SEND_OK:
    return false;
  case NetworkResultType::NET_RECEIVE_MSG:
    ProcessReceiveOK(msg);
    break;
  case NetworkResultType::NET_READ_OK:
    ProcessReadOK(msg);
    break;
  }
  return true;
}

void NetworkMessageQueue::ProcessReceiveOK(MsgPtr& msg) {
  auto& net_msg = result_.net_msg;
  // There is only one expected type of message: MSG_TYPE_REQUEST_ACK
  CHECK(net_msg.type == NetworkMessageType::MSG_TYPE_REQUEST_ACK);
  *msg = net_msg.actor_msg;
}

void NetworkMessageQueue::ProcessReadOK(MsgPtr& msg) {
  *msg = result_.net_msg.actor_msg;
}
}  // namespace oneflow
