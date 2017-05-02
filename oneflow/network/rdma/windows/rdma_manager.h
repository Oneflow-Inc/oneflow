#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_

#include <stdio.h>
#include <ndspi.h>

namespace oneflow{

class RdmaManager {

 public:
  RdmaManager();
  ~RdmaManager();

  bool Init();
  bool Destroy();

 private:
  sockaddr_in my_sock_;
  
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
  bool CreateConnector(Connection* conn);
  bool CreateQueuePair(Connection* conn);
  uint64_t WaitForConnection();

};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
