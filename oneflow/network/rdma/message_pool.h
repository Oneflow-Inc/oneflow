#ifndef ONEFLOW_NETWORK_RDMA_MESSAGE_POOL_H_
#define ONEFLOW_NETWORK_RDMA_MESSAGE_POOL_H_

#include <queue>
#include "network/network.h"
#include "network/rdma/switch.h"

namespace oneflow {

// Not thread-safe
template <typename MessageType>
class MessagePool {
public:
    explicit MessagePool(int32_t initial_size);
    ~MessagePool();

    MessageType* Alloc();
    void Free(MessageType* buffer);

private:
    std::queue<MessageType*> empty_buffer_;
};


}  // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_MESSAGE_POOL_H_
