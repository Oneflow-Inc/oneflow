#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_CONNECTION_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_CONNECTION_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_connection_info.pb.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

class IBVerbsConnection {
 public:
  explicit IBVerbsConnection() : qp_ptr_(nullptr) {}
  ~IBVerbsConnection() {
    if (qp_ptr_ != nullptr) { CHECK_EQ(ibv_destroy_qp(qp_ptr_), 0); }
  }

  void set_ibv_mtu(enum ibv_mtu active_mtu) { active_mtu_ = active_mtu; }
  void set_ibv_qp_ptr(ibv_qp* ibv_qp_ptr) { qp_ptr_ = ibv_qp_ptr; }
  IBVerbsConnectionInfo& mut_this_machine_conn_info() {
    return this_machine_conn_info_;
  }
  IBVerbsConnectionInfo& mut_peer_machine_conn_info() {
    return peer_machine_conn_info_;
  }

  IBVerbsConnectionInfo* mut_this_machine_conn_info_ptr() {
    return &this_machine_conn_info_;
  }
  IBVerbsConnectionInfo* mut_peer_machine_conn_info_ptr() {
    return &peer_machine_conn_info_;
  }

  void PostReadRequest(void* read_ctx, IBVerbsMemDesc* local_mem,
                       IBVerbsMemDescProto& remote_mem);
  void PostSendRequest(ActorMsg* msg, IBVerbsMemDesc* msg_mem);
  void PostRecvRequest(ActorMsg* msg, IBVerbsMemDesc* msg_mem);
  void CompleteConnection();

 private:
  IBVerbsConnectionInfo this_machine_conn_info_;
  IBVerbsConnectionInfo peer_machine_conn_info_;
  enum ibv_mtu active_mtu_;
  ibv_qp* qp_ptr_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_CONNECTION_H_
