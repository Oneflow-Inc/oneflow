#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MEMORY_DESC_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MEMORY_DESC_H_

#include "oneflow/core/comm_network/epoll/socket_memory_desc.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

struct SocketMemDesc {
  void* mem_ptr;
  size_t byte_size;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MEMORY_DESC_H_
