#ifndef ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_CONNECTION_H_
#define ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_CONNECTION_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <tchar.h>
#include <process.h>
#include <sal.h>
#include <new>
// #include <unistd.h>
#include <cstdint>
#include "oneflow/core/network/rdma/windows/ndsupport.h"
#include "oneflow/core/network/rdma/windows/ndcommon.h"
#include "oneflow/core/network/rdma/windows/interface.h"
#include "oneflow/core/network/network_memory.h"


namespace oneflow {

struct Request;
class RdmaMemory;

class Connection {
 public:
  explicit Connection(uint64_t my_machine_id);
  Connection(uint64_t my_machine_id, uint64_t peer_machine_id);
  ~Connection();

  bool Bind(const char* my_address, int port);
  bool TryConnectTo(const char* peer_address, int port);
  void CompleteConnectionTo();
  void AcceptConnect();

  void DestroyConnection();

  void PostSendRequest(Request* send_request);
  void PostRecvRequest(Request* recv_request);
  void PostReadRequest(Request* read_request,
                       MemoryDescriptor* remote_memory_descriptor,
                       RdmaMemory* dst_memory);

  void set_connector(IND2Connector* connector) { connector_ = connector; }
  void set_queue_pair(IND2QueuePair* queue_pair) { queue_pair_ = queue_pair; }
  void set_overlapped(OVERLAPPED* ov) { ov_ = ov; }

  IND2Connector* connector() { return connector_; }
  IND2QueuePair* queue_pair() { return queue_pair_; }
  OVERLAPPED* overlapped() { return ov_; }

 private:
  IND2Connector* connector_;
  IND2QueuePair* queue_pair_;
  OVERLAPPED* ov_;

  uint64_t my_machine_id_;
  uint64_t peer_machine_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_CONNECTION_H_
