#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_

// #include "network/rdma/windows/connection.h"

#include <stdint.h>
#include "ndspi.h"
#include "network/rdma/windows/interface.h"

namespace oneflow {

class Connection;
class Memory;

class RdmaManager {
 public:
  RdmaManager();
  ~RdmaManager();
  RdmaManager(const char* addr, int port);

  bool Init();
  bool Destroy();

  uint64_t WaitForConnection(Connection* conn);

  sockaddr_in my_sock;
  bool CreateConnector(Connection* conn);
  bool CreateQueuePair(Connection* conn);
  // uint64_t WaitForConnection();

  Memory* NewNetworkMemory();

 private:
  // NdspiV2 specific adatper and information
  IND2Adapter* adapter_;
  ND2_ADAPTER_INFO adapter_info_;
  HANDLE overlapped_file_;

  // Shared completion queue by all connections
  IND2CompletionQueue* send_cq_;  // send cq
  IND2CompletionQueue* recv_cq_;  // recv cq

  // Listener
  IND2Listener* listener_;

  bool InitAdapter();
  bool InitEnv();
};

}  // namespace oneflow

#endif  // ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
