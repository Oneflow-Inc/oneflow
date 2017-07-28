#ifndef ONEFLOW_CORE_NETWORK_RDMA_VERBS_RDMA_WRAPPER_H_
#define ONEFLOW_CORE_NETWORK_RDMA_VERBS_RDMA_WRAPPER_H_

#include <infiniband/verbs.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>

#include "oneflow/core/network/network_message.h"
#include "oneflow/core/network/rdma/verbs/interface.h"

namespace oneflow {

struct Request;

class RdmaWrapper {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaWrapper);
  RdmaWrapper();
  ~RdmaWrapper();

  void Init(const char* addr, int port);
  void Destroy();

  void CreateConnector(Connection* conn);
  void CreateQueuePair(Connection* conn);

  RdmaMemory* NewNetworkMemory();

  int64_t WaitForConnection(Connection* conn, Request* receive_request);

  int32_t PollRecvQueue(NetworkResult* result);
  int32_t PollSendQueue(NetworkResult* result);

 private:
  int my_sock_;
  sockaddr_in my_addr_;

  struct ibv_context* context_;  // TODO(shiyuan)
  struct ibv_pd* protect_domain_;  // TODO(shiyuan)
  // completion queue
  struct ibv_cq* send_cq_;  // TODO(shiyuan)
  struct ibv_cq* recv_cq_;  // TODO(shiyuan)

  struct rdma_cm_id* listener_;  // TODO(shiyuan)
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_VERBS_RDMA_WRAPPER_H_
