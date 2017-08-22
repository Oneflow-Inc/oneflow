#ifndef ONEFLOW_CORE_NETWORK_RDMA_LINUX_CONNECTION_H_
#define ONEFLOW_CORE_NETWORK_RDMA_LINUX_CONNECTION_H_

#include <infiniband/verbs.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/network/rdma/linux/rdma_memory.h"

namespace oneflow {

struct Request;

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
  ~Connection();

  void Bind(const std::string& my_ip, int32_t my_port);
  bool TryConnectTo(const std::string& peer_ip, int32_t peer_port);
  void CompleteConnection();
  void AcceptConnect();

  void PostSendRequest(const Request& send_request);
  void PostRecvRequest(const Request& recv_request);
  void PostReadRequest(const Request& read_request,
                       const MemoryDescriptor& remote_memory_descriptor,
                       RdmaMemory* dst_memory);

  void Destroy();

  Connector* connector() { return connector_; }
  ibv_qp* queue_pair() { return queue_pair_; }

  void set_connector(Connector* connector) { connector_ = connector; }
  void set_queue_pair(ibv_qp* queue_pair) {
    queue_pair_ = queue_pair;
    connector_->my_qpn = queue_pair_->qp_num;
  }

 private:
  void TransQueuePairState();

  int64_t my_machine_id_;
  Connector* connector_;
  ibv_qp* queue_pair_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_LINUX_CONNECTION_H_
