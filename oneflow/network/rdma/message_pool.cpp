#include "network/rdma/message_pool.h"

#include "network/network.h"

namespace oneflow {

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

} // namespace oneflow
