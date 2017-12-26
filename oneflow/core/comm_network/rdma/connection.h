#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H

#ifdef WITH_RDMA

#include <infiniband/verbs.h>
#include "glog/logging.h"
#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/comm_network/rdma/rdma_connection_info.pb.h"
#include "oneflow/core/comm_network/rdma/rdma_memory.h"

namespace oneflow {

class Connection {
 public:
  explicit Connection() : qp_ptr_(nullptr) {}
  ~Connection() {
    if (qp_ptr_ != nullptr) { CHECK_EQ(ibv_destroy_qp(qp_ptr_), 0); }
  }

  void set_ibv_mtu(enum ibv_mtu active_mtu) { active_mtu_ = active_mtu; }
  void set_ibv_qp_ptr(ibv_qp* ibv_qp_ptr) { qp_ptr_ = ibv_qp_ptr; }
  RdmaConnectionInfo& mut_peer_machine_conn_info() {
    return peer_machine_conn_info_;
  }
  RdmaConnectionInfo& mut_this_machine_conn_info() {
    return this_machine_conn_info_;
  }

  RdmaConnectionInfo* mut_this_machine_conn_info_ptr() {
    return &this_machine_conn_info_;
  }
  RdmaConnectionInfo* mut_peer_machine_conn_info_ptr() {
    return &peer_machine_conn_info_;
  }

  void PostReadRequest(void* read_ctx, const RdmaMem* local_mem,
                       const RdmaMemDesc& remote_mem);
  void PostSendRequest(const ActorMsg* msg, const RdmaMem* msg_mem);
  void PostRecvRequest(const ActorMsg* msg, const RdmaMem* msg_mem);
  void CompleteConnection();

 private:
  RdmaConnectionInfo peer_machine_conn_info_;
  RdmaConnectionInfo this_machine_conn_info_;
  enum ibv_mtu active_mtu_;
  ibv_qp* qp_ptr_;
};

}  // namespace oneflow

#endif  // WITH_RDMA

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H
