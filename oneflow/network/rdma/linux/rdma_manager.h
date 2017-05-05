#ifndef ONEFLOW_NETWORK_RDMA_LINUX_RDMA_MANAGER_H_
#define ONEFLOW_NETWORK_RDMA_LINUX_RDMA_MANAGER_H_

#include <stdint.h>
#include <stdio.h>
#include <rdma/rdma_cma.h>

#include "network/rdma/linux/interface.h"
#include "network/network_message.h"

namespace oneflow{

class Connection;
class Memory;

class RdmaManager {

public:
  struct rdm_cm_id* id;
  struct addrinfo* addr;  // TODO(shiyuan)


  RdmaManager();
  ~RdmaManager();

  bool Init(const char* addr, int port);
  bool Destroy();

  bool CreateConnector(Connection* conn);
  bool CreateQueuePair(connection* conn);

  Memory* NewNetworkMemory();

  uint64_t WaitForConnection(Connection* conn);

  int32_t PollRecvQueue(NetworkResult* result);
  int32_t PollSendQueue(NetworkResult* result);

  addr my_sock;  // TODO(shiyuan)

private:
  bool InitAdapter();
  bool InitEnv();
  
  struct rdma_cm_id* id_;
  struct addrinfo* addr;
  struct ibv_context ctx_;
  struct rdma_cm_event* event_;
  struct rdma_event_channel* ec_;
  struct ibv_pd* pd_;
  struct ibv_cq* recv_cq_;
  struct ibv_cq* send_cq_;
  struct ibv_comp_channel* comp_channel_;
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_LINUX_RDMA_MANAGER_H_
