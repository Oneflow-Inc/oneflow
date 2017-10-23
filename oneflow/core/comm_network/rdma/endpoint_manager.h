#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H

#include <infiniband/verbs.h>
#include <netdb.h>
#include <string>
#include "oneflow/core/comm_network/rdma/conn_info.pb.h"
#include "oneflow/core/comm_network/rdma/connection.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"

namespace oneflow {

class EndpointManager {
 public:
  void Init(const std::string& my_ip, int32_t my_port);
  RdmaMem* NewRdmaMem();
  Connection* NewConnection();
  ibv_qp* NewQueuePair();

  ConnectionInfo& GetMachineConnInfo() { return conn_info_; }

 private:
  void PollLoop();
  bool PollSendQueue();
  bool PollRecvQueue();

  ConnectionInfo conn_info_;
  ibv_context* context_;
  enum ibv_mtu active_mtu_;
  ibv_pd* pd_;
  ibv_cq* send_cq_;
  ibv_cq* recv_cq_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_ENDPOINT_MANAGER_H
