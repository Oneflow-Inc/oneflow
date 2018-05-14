#include "oneflow/core/comm_network/ibverbs/ibverbs_qp.h"
#include "oneflow/core/actor/actor_message_bus.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

IBVerbsQP::IBVerbsQP(ibv_context* ctx, ibv_pd* pd, ibv_cq* send_cq, ibv_cq* recv_cq) {
  // ctx_, pd_
  ctx_ = ctx;
  pd_ = pd;
  // max_recv_wr_
  ibv_device_attr device_attr;
  CHECK_EQ(ibv_query_device(ctx, &device_attr), 0);
  max_recv_wr_ = Global<JobDesc>::Get()->rdma_recv_msg_buf_byte() / sizeof(ActorMsg);
  max_recv_wr_ = std::min<uint32_t>(max_recv_wr_, device_attr.max_qp_wr);
  // qp_
  ibv_qp_init_attr qp_init_attr;
  qp_init_attr.qp_context = nullptr;
  qp_init_attr.send_cq = send_cq;
  qp_init_attr.recv_cq = recv_cq;
  qp_init_attr.srq = nullptr;
  qp_init_attr.cap.max_send_wr = device_attr.max_qp_wr;
  qp_init_attr.cap.max_recv_wr = max_recv_wr_;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.cap.max_inline_data = 0;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.sq_sig_all = 1;
  qp_ = ibv_create_qp(pd, &qp_init_attr);
  CHECK(qp_);
  // other
  remaining_sendable_wr_cnt_ = device_attr.max_qp_wr;
  CHECK(send_msg_buf_.empty());
  recv_msg_buf_.assign(max_recv_wr_, nullptr);
  FOR_RANGE(size_t, i, 0, recv_msg_buf_.size()) { recv_msg_buf_.at(i) = new ActorMsgMR(pd_); }
  remote_remaining_recvable_wr_cnt_ = 0;
  CHECK(pending_read_wr_.empty());
  CHECK(pending_send_wr_.empty());
}

IBVerbsQP::~IBVerbsQP() {
  CHECK(pending_send_wr_.empty());
  CHECK(pending_read_wr_.empty());
  CHECK_EQ(ibv_destroy_qp(qp_), 0);
  for (ActorMsgMR* msg_mr : recv_msg_buf_) { delete msg_mr; }
  while (send_msg_buf_.empty() == false) {
    delete send_msg_buf_.front();
    send_msg_buf_.pop();
  }
}

void IBVerbsQP::Connect(const IBVerbsConnectionInfo& peer_info) {
  ibv_port_attr port_attr;
  CHECK_EQ(ibv_query_port(ctx_, 1, &port_attr), 0);
  ibv_qp_attr qp_attr;
  // IBV_QPS_INIT
  memset(&qp_attr, 0, sizeof(ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = 1;
  qp_attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  CHECK_EQ(ibv_modify_qp(qp_, &qp_attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
           0);
  // IBV_QPS_RTR
  memset(&qp_attr, 0, sizeof(ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.ah_attr.grh.dgid.global.subnet_prefix = peer_info.subnet_prefix();
  qp_attr.ah_attr.grh.dgid.global.interface_id = peer_info.interface_id();
  qp_attr.ah_attr.grh.flow_label = 0;
  qp_attr.ah_attr.grh.sgid_index = 0;
  qp_attr.ah_attr.grh.hop_limit = std::numeric_limits<uint8_t>::max();
  qp_attr.ah_attr.dlid = peer_info.lid();
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.static_rate = 0;
  qp_attr.ah_attr.is_global = 1;
  qp_attr.ah_attr.port_num = 1;
  qp_attr.path_mtu = port_attr.active_mtu;
  qp_attr.dest_qp_num = peer_info.qp_num();
  qp_attr.rq_psn = 0;
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 12;
  CHECK_EQ(ibv_modify_qp(qp_, &qp_attr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN
                             | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
           0);
  // IBV_QPS_RTS
  memset(&qp_attr, 0, sizeof(ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.sq_psn = 0;
  qp_attr.max_rd_atomic = 1;
  qp_attr.retry_cnt = 7;
  qp_attr.rnr_retry = 7;
  qp_attr.timeout = 14;
  CHECK_EQ(ibv_modify_qp(qp_, &qp_attr,
                         IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_RETRY_CNT
                             | IBV_QP_RNR_RETRY | IBV_QP_TIMEOUT),

           0);
  // other
  for (ActorMsgMR* msg_mr : recv_msg_buf_) { PostRecvRequest(msg_mr); }
  remote_remaining_recvable_wr_cnt_ = peer_info.max_recv_wr();
}

void IBVerbsQP::PostReadRequest(const IBVerbsMemDescProto& remote_mem,
                                const IBVerbsMemDesc& local_mem, void* read_ctx) {
  FOR_RANGE(size_t, i, 0, local_mem.sge_vec().size()) {
    ibv_send_wr wr;
    wr.wr_id = reinterpret_cast<uint64_t>(read_ctx);  // TODO: add val in read_ctx
    wr.next = nullptr;
    wr.sg_list = const_cast<ibv_sge*>(&(local_mem.sge_vec().at(i)));
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = 0;
    wr.imm_data = 0;
    wr.wr.rdma.remote_addr = remote_mem.mem_ptr(i);
    wr.wr.rdma.rkey = remote_mem.mr_rkey(i);
    if (remaining_sendable_wr_cnt_ > 0) {
      ibv_send_wr* bad_wr = nullptr;
      CHECK_EQ(ibv_post_send(qp_, &wr, &bad_wr), 0);
      remaining_sendable_wr_cnt_ -= 1;
    } else {
      pending_read_wr_.push(wr);
    }
  }
}

void IBVerbsQP::PostSendRequest(const ActorMsg& msg) {
  ActorMsgMR* msg_mr = GetOneSendMsgMRFromBuf();
  msg_mr->set_msg(msg);
  ibv_send_wr wr;
  wr.wr_id = reinterpret_cast<uint64_t>(msg_mr);
  wr.next = nullptr;
  wr.sg_list = const_cast<ibv_sge*>(&(msg_mr->mem_desc().sge_vec().at(0)));
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = 0;
  wr.imm_data = 0;
  memset(&(wr.wr), 0, sizeof(wr.wr));
  if (remaining_sendable_wr_cnt_ > 0 && remote_remaining_recvable_wr_cnt_ > 0) {
    // TODO: (remote_remaining_recvable_wr_cnt_ > 0) != (remote has post the recv request)
    ibv_send_wr* bad_wr = nullptr;
    CHECK_EQ(ibv_post_send(qp_, &wr, &bad_wr), 0);
    remaining_sendable_wr_cnt_ -= 1;
    remote_remaining_recvable_wr_cnt_ -= 1;
  } else {
    pending_send_wr_.push(wr);
  }
}

void IBVerbsQP::ReadDone(uint64_t wr_id) {
  // TODO: for read_ctx
  remaining_sendable_wr_cnt_ += 1;
  TryPostPendingReadWR();
  TryPostPendingSendWR();
}

void IBVerbsQP::SendDone(uint64_t wr_id) {
  ActorMsgMR* msg_mr = reinterpret_cast<ActorMsgMR*>(wr_id);
  send_msg_buf_.push(msg_mr);
  remaining_sendable_wr_cnt_ += 1;
  remote_remaining_recvable_wr_cnt_ += 1;
  TryPostPendingSendWR();
  TryPostPendingReadWR();
}

void IBVerbsQP::RecvDone(uint64_t wr_id) {
  ActorMsgMR* msg_mr = reinterpret_cast<ActorMsgMR*>(wr_id);
  Global<ActorMsgBus>::Get()->SendMsgWithoutCommNet(msg_mr->msg());
  PostRecvRequest(msg_mr);
}

void IBVerbsQP::PostRecvRequest(ActorMsgMR* msg_mr) {
  ibv_recv_wr wr;
  wr.wr_id = reinterpret_cast<uint64_t>(msg_mr);
  wr.next = nullptr;
  wr.sg_list = const_cast<ibv_sge*>(&(msg_mr->mem_desc().sge_vec().at(0)));
  wr.num_sge = 1;
  ibv_recv_wr* bad_wr = nullptr;
  CHECK_EQ(ibv_post_recv(qp_, &wr, &bad_wr), 0);
}

ActorMsgMR* IBVerbsQP::GetOneSendMsgMRFromBuf() {
  if (send_msg_buf_.empty()) { send_msg_buf_.push(new ActorMsgMR(pd_)); }
  return send_msg_buf_.front();
}

void IBVerbsQP::TryPostPendingReadWR() {
  while (pending_read_wr_.empty() == false && remaining_sendable_wr_cnt_ > 0) {
    ibv_send_wr* bad_wr = nullptr;
    CHECK_EQ(ibv_post_send(qp_, &(pending_read_wr_.front()), &bad_wr), 0);
    pending_read_wr_.pop();
    remaining_sendable_wr_cnt_ -= 1;
  }
}

void IBVerbsQP::TryPostPendingSendWR() {
  while (pending_send_wr_.empty() == false && remaining_sendable_wr_cnt_ > 0
         && remote_remaining_recvable_wr_cnt_ > 0) {
    ibv_send_wr* bad_wr = nullptr;
    CHECK_EQ(ibv_post_send(qp_, &(pending_send_wr_.front()), &bad_wr), 0);
    pending_send_wr_.pop();
    remaining_sendable_wr_cnt_ -= 1;
    remote_remaining_recvable_wr_cnt_ -= 1;
  }
}

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX
