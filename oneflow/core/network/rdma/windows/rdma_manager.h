#ifndef ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
#define ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_

#include <stdint.h>
#include <ndspi.h>
#include "oneflow/core/network/rdma/windows/interface.h"
#include "oneflow/core/network/network_message.h"

namespace oneflow {

class Connection;
class RdmaMemory;
struct Request;

class RdmaManager {
 public:
  RdmaManager();
  ~RdmaManager();

  bool Init(const char* addr, int port);
  bool Destroy();

  bool CreateConnector(Connection* conn);
  bool CreateProtectDomain(Connection* conn);
  bool CreateQueuePair(Connection* conn);

  RdmaMemory* NewNetworkMemory();

  uint64_t WaitForConnection(Connection* conn, Request* receive_request);

  int32_t PollRecvQueue(NetworkResult* result);
  int32_t PollSendQueue(NetworkResult* result);

  void set_my_sock(sockaddr_in my_sock) { my_sock_ = my_sock; }
  sockaddr_in my_sock() { return my_sock_; }

 private:
  sockaddr_in my_sock_;

  // NdspiV2 specific adatper and information
  IND2Adapter* adapter_;
  ND2_ADAPTER_INFO adapter_info_;
  HANDLE overlapped_file_;

  // completion queue
  IND2CompletionQueue* send_cq_;  // send cq
  IND2CompletionQueue* recv_cq_;  // recv cq

  // Listener
  IND2Listener* listener_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_RDMA_MANAGER_H_
