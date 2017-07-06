#ifndef ONEFLOW_CORE_NETWORK_RDMA_LINUX_CONNECTION_H_
#define ONEFLOW_CORE_NETWORK_RDMA_LINUX_CONNECTION_H_

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/network/rdma/linux/interface.h"

namespace oneflow {

struct Request;
class RdmaMemory;

struct Connector {
  int32_t my_lid;
  int32_t my_qpn;
  int32_t my_psn;
  uint64_t my_snp;
  uint64_t my_iid;
  int32_t peer_lid;
  int32_t peer_qpn;
  int32_t peer_psn;
  uint64_t peer_snp;
  uint64_t peer_iid;
  enum ibv_mtu active_mtu;
};

class Connection {
 public:
  explicit Connection(int64_t my_machine_id);
  Connection(int64_t my_machine_id, int64_t peer_machine_id);
  ~Connection();

  void Bind(const char* my_address, int port);
  void TryConnectTo(const char* peer_address, int port);
  void CompleteConnectionTo();
  void AcceptConnect();

  void DestroyConnection();

  void PostSendRequest(Request* send_request);
  void PostRecvRequest(Request* recv_request);
  void PostReadRequest(Request* read_request,
                       MemoryDescriptor* remote_memory_descriptor,
                       RdmaMemory* dst_memory);

  void set_connector(struct Connector* connector) { connector_ = connector; }
  void set_queue_pair(struct ibv_qp* queue_pair) {
    queue_pair_ = queue_pair;
    connector_->my_qpn = queue_pair_->qp_num;
  }

  struct Connector* connector() {
    return connector_;
  }
  struct ibv_qp* queue_pair() {
    return queue_pair_;
  }

 private:
  struct Connector* connector_;
  struct ibv_qp* queue_pair_;

  int64_t my_machine_id_;
  int64_t peer_machine_id_;

  struct sockaddr_in my_addr_;
  int my_sock_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_LINUX_CONNECTION_H_
