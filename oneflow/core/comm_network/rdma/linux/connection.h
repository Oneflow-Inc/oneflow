#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_LINUX_CONNECTION_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_LINUX_CONNECTION_H

#include "oneflow/core/comm_network/rdma/rdma/rdma_memory.h"

namespace oneflow {

class Connection {
public:
  void PostSendRequest(
      void* read_ctx, RdmaMem* local_mem, const RdmaMemDesc& remote_mem);
  void PostReadRequest(RdmaMem* msg_mem);
  void PostRecvRequest(RdmaMem* msg_mem);
};

}

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_LINUX_CONNECTION_H
