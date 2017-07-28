#ifndef ONEFLOW_CORE_NETWORK_RDMA_VERBS_RDMA_MESSAGE_H_
#define ONEFLOW_CORE_NETWORK_RDMA_VERBS_RDMA_MESSAGE_H_

#include "oneflow/core/network/network.h"
#include "oneflow/core/network/network_message.h"
#include "oneflow/core/network/rdma/verbs/rdma_memory.h"

namespace oneflow {

class RdmaMemory;

class RdmaMessage {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaMessage);
  RdmaMessage();
  ~RdmaMessage();

  const NetworkMessage& msg() const { return net_msg_; }
  NetworkMessage& mutable_msg() { return net_msg_; }
  NetworkMemory* net_memory() { return net_memory_; }

 private:
  NetworkMessage net_msg_;
  NetworkMemory* net_memory_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_VERBS_RDMA_MESSAGE_H_
