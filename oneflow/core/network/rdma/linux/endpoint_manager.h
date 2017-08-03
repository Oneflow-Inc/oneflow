#ifndef ONEFLOW_CORE_NETWORK_RDMA_LINUX_ENDPOINT_MANAGER_H_
#define ONEFLOW_CORE_NETWORK_RDMA_LINUX_ENDPOINT_MANAGER_H_

#include <infiniband/verbs.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>

#include "oneflow/core/network/network_message.h"
#include "oneflow/core/network/rdma/linux/interface.h"

namespace oneflow {

class Connection;
struct Request;

class EndpointManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EndpointManager);
  EndpointManager() = default;
  ~EndpointManager();

  void Init(const char* my_ip, int32_t my_port);
  void Destroy();

  void CreateConnector(Connection* conn);
  void CreateQueuePair(Connection* conn);

  RdmaMemory* NewNetworkMemory();

  int64_t WaitForConnection(Connection* conn, Request* receive_request);

  Request* PollRecvQueue(NetworkResult* result);
  Request* PollSendQueue(NetworkResult* result);

 private:
  int32_t my_sock_;
  sockaddr_in my_addr_;

  ibv_context* context_;
  ibv_pd* protect_domain_;

  // completion queue
  ibv_cq* send_cq_;
  ibv_cq* recv_cq_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_LINUX_ENDPOINT_MANAGER_H_
