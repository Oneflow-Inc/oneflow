#ifndef ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_CONNECTION_H_
#define ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_CONNECTION_H_

#include <process.h>
#include <sal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>
#include <cstdint>
#include <new>
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/network/rdma/windows/rdma_memory.h"

namespace oneflow {

struct Request;

class Connection {
 public:
  explicit Connection(int64_t my_machine_id);
  Connection(int64_t my_machine_id, int64_t peer_machine_id);
  ~Connection();

  void Bind(const char* my_ip, int32_t my_port);
  bool TryConnectTo(const char* peer_ip, int32_t peer_port);
  void CompleteConnection();
  void AcceptConnect();

  void PostSendRequest(const Request& send_request);
  void PostRecvRequest(const Request& recv_request);
  void PostReadRequest(const Request& read_request,
                       const MemoryDescriptor& remote_memory_descriptor,
                       RdmaMemory* dst_memory);

  void Destroy();

  IND2Connector* mutable_connector() { return connector_; }
  IND2QueuePair* mutable_queue_pair() { return queue_pair_; }
  OVERLAPPED* mutable_overlapped() { return ov_; }

  void set_connector(IND2Connector* connector);
  void set_queue_pair(IND2QueuePair* queue_pair);
  void set_overlapped(OVERLAPPED* ov);

  const IND2Connector& connector() { return *connector_; }
  const IND2QueuePair& queue_pair() { return *queue_pair_; }
  const OVERLAPPED& overlapped() { return *ov_; }

 private:
  int64_t my_machine_id_;
  int64_t peer_machine_id_;

  IND2Connector* connector_;
  IND2QueuePair* queue_pair_;
  OVERLAPPED* ov_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_WINDOWS_CONNECTION_H_
