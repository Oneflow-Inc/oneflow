#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H

#include <infiniband/verbs.h>
#include <netdb.h>
#include <string>
#include "oneflow/core/comm_network/rdma/connection.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"

namespace oneflow {

class EndpointManager {
 public:
  void Init(const std::string& my_ip, int32_t my_port);
  RdmaMem* NewRdmaMem();
  Connector* NewConnector();
  ibv_qp* NewQueuePair();

 private:
  void PollLoop();
  bool PollSendQueue();
  bool PollRecvQueue();

  ibv_context* context_;
  ibv_pd* pd_;
  ibv_cq* send_cq_;
  ibv_cq* recv_cq_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H
