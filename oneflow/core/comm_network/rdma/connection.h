#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H

#include <infiniband/verbs.h>
#include "glog/logging.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"

namespace oneflow {

struct ConnectorInfo {
  int32_t lid;
  int32_t qpn;
  uint32_t snp;
  uint32_t iid;
};

struct Connector {
  ConnectorInfo my_conn_info;
  ConnectorInfo peer_conn_info;
  enum ibv_mtu active_mtu;
};

class Connection {
 public:
  explicit Connection();
  ~Connection();

  void set_my_connector(const ConnectorInfo& my_conn_info) {
    conn_->my_conn_info.lid = my_conn_info.lid;
    conn_->my_conn_info.qpn = my_conn_info.qpn;
    conn_->my_conn_info.snp = my_conn_info.snp;
    conn_->my_conn_info.iid = my_conn_info.iid;
  }
  void set_peer_connector(const ConnectorInfo& peer_conn_info) {
    conn_->peer_conn_info.lid = peer_conn_info.lid;
    conn_->peer_conn_info.qpn = peer_conn_info.qpn;
    conn_->peer_conn_info.snp = peer_conn_info.snp;
    conn_->peer_conn_info.iid = peer_conn_info.iid;
  }

  void ConnectTo(int64_t peer_machine_id);

  void PostReadRequest(void* read_ctx, const RdmaMem* local_mem,
                       const RdmaMemDesc* remote_mem);
  void PostSendRequest(const RdmaMem* msg_mem);
  void PostRecvRequest(const RdmaMem* msg_mem);
  void WaitForConnection();

 private:
  Connector* conn_;
  ibv_qp* qp_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H
