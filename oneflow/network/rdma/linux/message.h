#ifndef ONEFLOW_NETWORK_RDMA_LINUX_MESSAGE_H_
#define ONEFLOW_NETWORK_RDMA_LINUX_MESSAGE_H_

#include "network/network.h"
#include "network/network_message.h"
#include "network/rdma/linux/memory.h"

namespace oneflow {

class Memory;

class Message {
public:
    Message();
    ~Message();

    const NetworkMessage& msg() const { return net_msg_; }
    NetworkMessage& mutable_msg() { return net_msg_; }
    NetworkMemory* net_memory() { return net_memory_; }

private:
    NetworkMessage net_msg_;
    NetworkMemory* net_memory_;
};

} // namespace oneflow


#endif // ONEFLOW_NETWORK_RDMA_LINUX_MESSAGE_H_ 
