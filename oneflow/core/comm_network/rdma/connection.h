#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H

#include <infiniband/verbs.h>
#include "glog/logging.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"
#include "oneflow/core/control/ctrl_client.h"

namespace oneflow {

struct ConnectorInfo {
  ConnectorInfo& operator=(ConnectorInfo& conn_info) {
    lid = conn_info.lid;
    qpn = conn_info.qpn;
    snp = conn_info.snp;
    iid = conn_info.iid;
  }
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

  void set_my_conn_info(const ConnectorInfo& my_conn_info) {
    conn_->my_conn_info = my_conn_info;
  }
  void set_peer_conn_info(const ConnectorInfo& peer_conn_info) {
    conn_->peer_conn_info = peer_conn_info;
  }

  void PostReadRequest(void* read_ctx, RdmaMem* local_mem,
                       RdmaMemDesc* remote_mem);
  void PostSendRequest(RdmaMem* msg_mem);
  void PostRecvRequest(RdmaMem* msg_mem);

 private:
  Connector* conn_;
  ibv_qp* qp_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H
