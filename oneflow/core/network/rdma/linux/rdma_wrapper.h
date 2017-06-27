#ifndef ONEFLOW_CORE_NETWORK_RDMA_LINUX_RDMA_WRAPPER_H_
#define ONEFLOW_CORE_NETWORK_RDMA_LINUX_RDMA_WRAPPER_H_

#include <stdint.h>
#include <stdio.h>
#include <netdb.h>
#include <infiniband/verbs.h>

#include "oneflow/core/network/rdma/linux/interface.h"
#include "oneflow/core/network/network_message.h"

namespace oneflow {

class Connection;
class RdmaMemory;
struct Request;

class RdmaWrapper {
 public:
  RdmaWrapper();
  ~RdmaWrapper();

  void Init(const char* addr, int port);
  void Destroy();

  void CreateConnector(Connection* conn);
  void CreateQueuePair(Connection* conn);

  RdmaMemory* NewNetworkMemory();

  uint64_t WaitForConnection(Connection* conn, Request* receive_request);

  int32_t PollRecvQueue(NetworkResult* result);
  int32_t PollSendQueue(NetworkResult* result);

 private:
  int my_sock_;
  sockaddr_in my_addr_;

  struct rdma_event_channel* ec_;
  struct ibv_context* context_;
  struct ibv_pd* protect_domain_;
  
  // completion queue
  struct ibv_cq* recv_cq_;
  struct ibv_cq* send_cq_;

  struct rdma_cm_id* listener_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_LINUX_RDMA_WRAPPER_H_
