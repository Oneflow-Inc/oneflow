#ifndef ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_ENDPOINT_MANAGER_H_
#define ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_ENDPOINT_MANAGER_H_

#include <stdint.h>
#include "oneflow/core/network/network_message.h"
#include "oneflow/core/network/rdma/windows/ndsupport.h"
#include "oneflow/core/network/rdma/windows/connection.h"
#include "oneflow/core/network/rdma/windows/rdma_memory.h"

namespace oneflow {

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

  int64_t WaitForConnection(Connection* conn);

  Request* PollRecvQueue(NetworkResult* result);
  Request* PollSendQueue(NetworkResult* result);

 private:
  sockaddr_in my_addr_;

  // NdspiV2 specific adatper and information
  IND2Adapter* adapter_;
  ND2_ADAPTER_INFO adapter_info_;
  HANDLE overlapped_file_;

  // completion queue
  IND2CompletionQueue* send_cq_;
  IND2CompletionQueue* recv_cq_;

  // Listener
  IND2Listener* listener_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_ENDPOINT_MANAGER_H_
