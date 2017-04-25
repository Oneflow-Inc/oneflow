#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_

#include <stdio.h>
#include <ndspi.h>

namespace oneflow{

class RdmaManager {

public:
  IND2Adapter* adapter_;
  sockaddr_in sin;

  RdmaManager();
  ~RdmaManager();

  bool Init();
  bool Destroy();

private:
  // NdspiV2 specific adatper and information
  ND2_ADAPTER_INFO adapter_info_;
  HANDLE overlapped_file_;
    
  IND2Listener* listener_;
    
  // Shared completion queue by all connections
  IND2CompletionQueue* send_cq_;  // send cq
  IND2CompletionQueue* recv_cq_;  // recv cq


  bool InitAdapter();
  bool InitEnv();
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
