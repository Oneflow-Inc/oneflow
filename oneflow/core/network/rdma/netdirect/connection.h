#ifndef ONEFLOW_CORE_NETWORK_RDMA_NETDIRECT_CONNECTION_H_
#define ONEFLOW_CORE_NETWORK_RDMA_NETDIRECT_CONNECTION_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <tchar.h>
#include <process.h>
#include <sal.h>
#include <new>
#include <cstdint>
#include "oneflow/core/network/rdma/netdirect/interface.h"
#include "oneflow/core/network/network_memory.h"

namespace oneflow {

struct Request;
class RdmaMemory;

class Connection {
 public:
  explicit Connection(int64_t my_machine_id);
  Connection(int64_t my_machine_id, int64_t peer_machine_id);
  ~Connection();

  void Bind(const char* my_address, int port);
  bool TryConnectTo(const char* peer_address, int port);
  void CompleteConnectionTo();
  void AcceptConnect();

  void DestroyConnection();

  void PostSendRequest(const Request& send_request);
  void PostRecvRequest(const Request& recv_request);
  void PostReadRequest(const Request& read_request,
                       const MemoryDescriptor& remote_memory_descriptor,
                       RdmaMemory* dst_memory);

  void set_connector(IND2Connector* connector) { connector_ = connector; }
  void set_queue_pair(IND2QueuePair* queue_pair) { queue_pair_ = queue_pair; }
  void set_overlapped(OVERLAPPED* ov) { ov_ = ov; }

  IND2Connector* connector() { return connector_; }
  IND2QueuePair* queue_pair() { return queue_pair_; }
  OVERLAPPED* overlapped() { return ov_; }

 private:
  IND2Connector* connector_;  // TODO(shiyuan)
  IND2QueuePair* queue_pair_;  // TODO(shiyuan)
  OVERLAPPED* ov_;  // TODO(shiyuan)

  int64_t my_machine_id_;
  int64_t peer_machine_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_NETDIRECT_CONNECTION_H_
