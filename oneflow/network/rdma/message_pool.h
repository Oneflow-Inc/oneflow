#ifndef ONEFLOW_NETWORK_RDMA_MESSAGE_POOL_H_
#define ONEFLOW_NETWORK_RDMA_MESSAGE_POOL_H_

#include <queue>
#include "network/network.h"
#include "network/rdma/memory.h" // 
//#include "network/rdma/registered_network_message.h" // move message to platform/common.h
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

template <typename MessageType>
MessagePool<MessageType>::MessagePool(int32_t initial_size) {
    for (int i = 0; i < initial_size; ++i) {
        empty_buffer_.push(new MessageType());
    }
}

template <typename MessageType>
MessagePool<MessageType>::~MessagePool() {
    while (!empty_buffer_.empty()) {
        MessageType* buffer = empty_buffer_.front();
        empty_buffer_.pop();
        delete buffer;
    }
}

template <typename MessageType>
MessageType* MessagePool<MessageType>::Alloc() {
    // NOTE(feiga): for easy implementation, we always allocate new
    //              if no available buffer.
    // Or we can return nullptr, implying no available buffer, then application
    // shall try to Alloc and Send later.
    if (empty_buffer_.empty()) {
        // TODO(jiyuan): verify whether the |new| succeeds
        empty_buffer_.push(new MessageType());
    }
    MessageType* buffer = empty_buffer_.front();
    empty_buffer_.pop();
    return buffer;
}

template <typename MessageType>
void MessagePool<MessageType>::Free(MessageType* buffer) {
    if (buffer != nullptr) empty_buffer_.push(buffer);
}

}  // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_MESSAGE_POOL_H_
