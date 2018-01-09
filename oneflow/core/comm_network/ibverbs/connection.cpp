#include "oneflow/core/comm_network/ibverbs/connection.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

void Connection::PostReadRequest(void* read_ctx, IBVerbsMemDesc* local_mem,
                                 IBVerbsMemDescProto& remote_mem) {
  ibv_send_wr wr, *bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(read_ctx);
  wr.opcode = IBV_WR_RDMA_READ;
  wr.sg_list = local_mem->ibv_sge_ptr();
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_mem.mem_ptr();
  wr.wr.rdma.rkey = remote_mem.mr_rkey();

  // return val may be incorrect when successfully executing
  ibv_post_send(qp_ptr_, &wr, &bad_wr);
}

void Connection::PostSendRequest(ActorMsg* msg, IBVerbsMemDesc* msg_mem) {
  ibv_send_wr wr, *bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(msg);
  wr.next = nullptr;
  wr.sg_list = msg_mem->ibv_sge_ptr();
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  CHECK_EQ(ibv_post_send(qp_ptr_, &wr, &bad_wr), 0);
}

void Connection::PostRecvRequest(ActorMsg* msg, IBVerbsMemDesc* msg_mem) {
  ibv_recv_wr wr, *bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(msg);
  wr.next = nullptr;
  wr.sg_list = msg_mem->ibv_sge_ptr();
  wr.num_sge = 1;

  CHECK_EQ(ibv_post_recv(qp_ptr_, &wr, &bad_wr), 0);
}

void Connection::CompleteConnection() {
  ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(ibv_qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = 1;
  qp_attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

  CHECK_EQ(ibv_modify_qp(qp_ptr_, &qp_attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT
                             | IBV_QP_ACCESS_FLAGS),
           0);

  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.path_mtu = active_mtu_;
  qp_attr.dest_qp_num = peer_machine_conn_info_.qpn();
  qp_attr.rq_psn = peer_machine_conn_info_.psn();
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 12;
  qp_attr.ah_attr.is_global = 1;
  qp_attr.ah_attr.grh.dgid.global.subnet_prefix = peer_machine_conn_info_.snp();
  qp_attr.ah_attr.grh.dgid.global.interface_id = peer_machine_conn_info_.iid();
  qp_attr.ah_attr.grh.flow_label = 0;
  qp_attr.ah_attr.grh.hop_limit = 255;
  qp_attr.ah_attr.dlid = peer_machine_conn_info_.lid();
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = 1;

  CHECK_EQ(
      ibv_modify_qp(qp_ptr_, &qp_attr,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN
                        | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC
                        | IBV_QP_MIN_RNR_TIMER),
      0);

  memset(&qp_attr, 0, sizeof(ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.sq_psn = this_machine_conn_info_.psn();
  qp_attr.timeout = 14;
  qp_attr.retry_cnt = 7;
  qp_attr.rnr_retry = 7;
  qp_attr.max_rd_atomic = 1;

  CHECK_EQ(ibv_modify_qp(qp_ptr_, &qp_attr,
                         IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT
                             | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN
                             | IBV_QP_MAX_QP_RD_ATOMIC),
           0);
}

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX
