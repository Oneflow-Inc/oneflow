#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H

#include <infiniband/verbs.h>
#include "glog/logging.h"
#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/comm_network/rdma/conn_info.pb.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"

namespace oneflow {

class Connection {
 public:
  explicit Connection();
  ~Connection();

  void set_ibv_mtu(enum ibv_mtu active_mtu) { active_mtu_ = active_mtu; }
  void set_ibv_qp_ptr(ibv_qp* ibv_qp_ptr) { qp_ptr_ = ibv_qp_ptr; }
  void set_peer_conn_info(ConnectionInfo& peer_conn_info) {
    peer_conn_info_ = peer_conn_info;
  }

  void PostReadRequest(void* read_ctx, const RdmaMem* local_mem,
                       const RdmaMemDesc* remote_mem);
  void PostSendRequest(const ActorMsg& msg, const RdmaMem* msg_mem);
  void PostRecvRequest(const ActorMsg& msg, const RdmaMem* msg_mem);
  void WaitForConnection();

 private:
  ConnectionInfo peer_conn_info_;
  enum ibv_mtu active_mtu_;
  ibv_qp* qp_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H
