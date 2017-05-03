#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_

// #include "network/rdma/windows/connection.h"

#include <stdint.h>
#include <ndspi.h>
#include "network/rdma/windows/interface.h"
#include "network\network_message.h"

namespace oneflow {

class Connection;
class Memory;

class RdmaManager {
 public:
  RdmaManager(const char* addr, int port);
  ~RdmaManager();

  bool Init();
  bool Destroy();

  bool CreateConnector(Connection* conn);
  bool CreateQueuePair(Connection* conn);

  Memory* NewNetworkMemory();
  
  uint64_t WaitForConnection(Connection* conn);
  
  int32_t PollRecvQueue(NetworkResult* result);
  int32_t PollSendQueue(NetworkResult* result);

  sockaddr_in my_sock;

 private:
  bool InitAdapter();
  bool InitEnv();
  
  // NdspiV2 specific adatper and information
  IND2Adapter* adapter_;
  ND2_ADAPTER_INFO adapter_info_;
  HANDLE overlapped_file_;

  // Shared completion queue by all connections
  IND2CompletionQueue* send_cq_;  // send cq
  IND2CompletionQueue* recv_cq_;  // recv cq

  // Listener
  IND2Listener* listener_;
};

}  // namespace oneflow

#endif  // ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
