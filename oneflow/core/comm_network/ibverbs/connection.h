#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H_
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H_

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/comm_network/ibverbs/rdma_connection_info.pb.h"
#include "oneflow/core/comm_network/ibverbs/rdma_memory.h"

namespace oneflow {

class Connection {
 public:
  explicit Connection() : qp_ptr_(nullptr) {}
  ~Connection() {
    if (qp_ptr_ != nullptr) { CHECK_EQ(ibv_destroy_qp(qp_ptr_), 0); }
  }

  void set_ibv_mtu(enum ibv_mtu active_mtu) { active_mtu_ = active_mtu; }
  void set_ibv_qp_ptr(ibv_qp* ibv_qp_ptr) { qp_ptr_ = ibv_qp_ptr; }
  RdmaConnectionInfo& mut_this_machine_conn_info() {
    return this_machine_conn_info_;
  }
  RdmaConnectionInfo& mut_peer_machine_conn_info() {
    return peer_machine_conn_info_;
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
  RdmaConnectionInfo this_machine_conn_info_;
  RdmaConnectionInfo peer_machine_conn_info_;
  enum ibv_mtu active_mtu_;
  ibv_qp* qp_ptr_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_CONNECTION_H_
