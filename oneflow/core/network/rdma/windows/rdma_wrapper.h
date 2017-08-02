#ifndef ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_RDMA_WRAPPER_H_
#define ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_RDMA_WRAPPER_H_

#include <stdint.h>
#include <ndspi.h>
#include "oneflow/core/network/rdma/windows/interface.h"
#include "oneflow/core/network/network_message.h"

namespace oneflow {

class Connection;
struct Request;

class RdmaWrapper {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RdmaWrapper);
  RdmaWrapper();
  ~RdmaWrapper();

  void Init(const char* my_ip, int32_t my_port);
  void Destroy();

  void CreateConnector(Connection* conn);
  void CreateQueuePair(Connection* conn);

  RdmaMemory* NewNetworkMemory();

  int64_t WaitForConnection(Connection* conn, Request* receive_request);

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

#endif  // ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_RDMA_WRAPPER_H_
