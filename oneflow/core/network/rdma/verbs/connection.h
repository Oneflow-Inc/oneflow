#ifndef ONEFLOW_CORE_NETWORK_RDMA_VERBS_CONNECTION_H_
#define ONEFLOW_CORE_NETWORK_RDMA_VERBS_CONNECTION_H_

#include <infiniband/verbs.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/network/rdma/verbs/interface.h"

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
  OF_DISALLOW_COPY_AND_MOVE(Connection);
  explicit Connection(int64_t my_machine_id);
  Connection(int64_t my_machine_id, int64_t peer_machine_id);
  ~Connection();

  // TODO(shiyuan) discuss these 4 function
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

  // TODO(shiyuan) memory manage
  void set_connector(Connector* connector) {
    if (connector_ != nullptr) {
      delete connector_;
    }
    connector_ = connector;
  }
  void set_queue_pair(ibv_qp* queue_pair) {
    if (queue_pair_ != nullptr) {
      delete queue_pair_;
    }
    queue_pair_ = queue_pair;
    connector_->my_qpn = queue_pair_->qp_num;
  }

  // TODO(shiyuan) call and like
  const Connector& connector() { return *connector_; }
  const ibv_qp& queue_pair() { return *queue_pair_; }

 private:
  std::unique_ptr<Connector> connector_;  // TODO(shiyuan)
  std::shared_ptr<ibv_qp> queue_pair_;  // TODO(shiyuan)

  int64_t my_machine_id_;
  int64_t peer_machine_id_;

  sockaddr_in my_addr_;
  int my_sock_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_RDMA_VERBS_CONNECTION_H_
