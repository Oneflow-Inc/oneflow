#include "oneflow/core/comm_network/ibverbs/ibverbs_qp.h"
#include "oneflow/core/actor/actor_message_bus.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

namespace {

void CheckIBVerbsConf(const IBVerbsConf& ibv_conf) {
  CHECK_GE(ibv_conf.pkey_index(), 0);
  CHECK_LE(ibv_conf.pkey_index(), GetMaxVal<uint8_t>());
  CHECK_GE(ibv_conf.timeout(), 0);
  CHECK_LE(ibv_conf.timeout(), GetMaxVal<uint8_t>());
  CHECK_GE(ibv_conf.retry_cnt(), 0);
  CHECK_LE(ibv_conf.retry_cnt(), GetMaxVal<uint8_t>());
  CHECK_GE(ibv_conf.sl(), 0);
  CHECK_LE(ibv_conf.sl(), 7);
}

}  // namespace

IBVerbsQP::IBVerbsQP(ibv_context* ctx, ibv_pd* pd, ibv_cq* cq)
    : ibv_conf_(Global<JobDesc>::Get()->ibverbs_conf()), ctx_(ctx), pd_(pd) {
  CheckIBVerbsConf(ibv_conf_);
  // qp_
  ibv_device_attr device_attr;
  PCHECK(ibv_query_device(ctx, &device_attr) == 0);
  ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.send_cq = cq;
  qp_init_attr.recv_cq = cq;
  qp_init_attr.cap.max_send_wr = ibv_conf_.queue_depth();
  qp_init_attr.cap.max_recv_wr = ibv_conf_.queue_depth();
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.cap.max_inline_data = 0;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.sq_sig_all = 1;  // all wrs submitted to the send queue will always generate a wc
  qp_ = ibv_create_qp(pd, &qp_init_attr);
  PCHECK(qp_);
  // recv_msg_buf_
  recv_msg_buf_.assign(ibv_conf_.queue_depth(), nullptr);
  FOR_RANGE(size_t, i, 0, recv_msg_buf_.size()) { recv_msg_buf_.at(i) = new ActorMsgMR(pd_); }
  // send_msg_buf_
  CHECK(send_msg_buf_.empty());
}

IBVerbsQP::~IBVerbsQP() {
  PCHECK(ibv_destroy_qp(qp_) == 0);
  while (send_msg_buf_.empty() == false) {
    delete send_msg_buf_.front();
    send_msg_buf_.pop();
  }
  for (ActorMsgMR* msg_mr : recv_msg_buf_) { delete msg_mr; }
}

void IBVerbsQP::Connect(const IBVerbsConnectionInfo& peer_info) {
  ibv_port_attr port_attr;
  PCHECK(ibv_query_port(ctx_, 1, &port_attr) == 0);
  ibv_qp_attr qp_attr;
  // IBV_QPS_INIT
  memset(&qp_attr, 0, sizeof(ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = ibv_conf_.pkey_index();
  qp_attr.port_num = ibv_conf_.port_num();
  qp_attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  PCHECK(ibv_modify_qp(qp_, &qp_attr,
                       IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)
         == 0);
  // IBV_QPS_RTR
  memset(&qp_attr, 0, sizeof(ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.ah_attr.grh.dgid.global.subnet_prefix = peer_info.subnet_prefix();
  qp_attr.ah_attr.grh.dgid.global.interface_id = peer_info.interface_id();
  qp_attr.ah_attr.grh.flow_label = 0;
  qp_attr.ah_attr.grh.sgid_index = ibv_conf_.sgid_index();
  qp_attr.ah_attr.grh.traffic_class = ibv_conf_.traffic_class();
  qp_attr.ah_attr.grh.hop_limit = GetMaxVal<uint8_t>();
  qp_attr.ah_attr.dlid = peer_info.local_id();
  qp_attr.ah_attr.sl = ibv_conf_.sl();
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.static_rate = 0;
  qp_attr.ah_attr.is_global = 1;
  qp_attr.ah_attr.port_num = ibv_conf_.port_num();
  qp_attr.path_mtu = port_attr.active_mtu;
  qp_attr.dest_qp_num = peer_info.qp_num();
  qp_attr.rq_psn = 0;
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 12;
  PCHECK(ibv_modify_qp(qp_, &qp_attr,
                       IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN
                           | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)
         == 0);
  // IBV_QPS_RTS
  memset(&qp_attr, 0, sizeof(ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.sq_psn = 0;  // TODO(shiyuan) static_cast<uint32_t>(random::New64()) & 0xffffff
  qp_attr.max_rd_atomic = 1;
  qp_attr.retry_cnt = ibv_conf_.retry_cnt();
  qp_attr.rnr_retry = 7;  // infinite
  qp_attr.timeout = ibv_conf_.timeout();
  PCHECK(ibv_modify_qp(qp_, &qp_attr,
                       IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_RETRY_CNT
                           | IBV_QP_RNR_RETRY | IBV_QP_TIMEOUT)
         == 0);
}

void IBVerbsQP::PostAllRecvRequest() {
  for (ActorMsgMR* msg_mr : recv_msg_buf_) { PostRecvRequest(msg_mr); }
}

void IBVerbsQP::PostReadRequest(const IBVerbsMemDescProto& remote_mem,
                                const IBVerbsMemDesc& local_mem, int64_t stream_id) {
  CHECK_EQ(remote_mem.mem_ptr_size(), local_mem.sge_vec().size());
  WorkRequestId* wr_id = NewWorkRequestId();
  wr_id->outstanding_sge_cnt = local_mem.sge_vec().size();
  wr_id->stream_id = stream_id;
  FOR_RANGE(size_t, i, 0, local_mem.sge_vec().size()) {
    ibv_send_wr wr;
    wr.wr_id = reinterpret_cast<uint64_t>(wr_id);
    wr.next = nullptr;
    wr.sg_list = const_cast<ibv_sge*>(&(local_mem.sge_vec().at(i)));
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = 0;
    wr.imm_data = 0;
    wr.wr.rdma.remote_addr = remote_mem.mem_ptr(i);
    wr.wr.rdma.rkey = remote_mem.mr_rkey(i);
    ibv_send_wr* bad_wr = nullptr;
    PCHECK(ibv_post_send(qp_, &wr, &bad_wr) == 0);
  }
}

void IBVerbsQP::PostSendRequest(const ActorMsg& msg) {
  ActorMsgMR* msg_mr = GetOneSendMsgMRFromBuf();
  msg_mr->set_msg(msg);
  WorkRequestId* wr_id = NewWorkRequestId();
  wr_id->msg_mr = msg_mr;
  ibv_send_wr wr;
  wr.wr_id = reinterpret_cast<uint64_t>(wr_id);
  wr.next = nullptr;
  wr.sg_list = const_cast<ibv_sge*>(&(msg_mr->mem_desc().sge_vec().at(0)));
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = 0;
  wr.imm_data = 0;
  memset(&(wr.wr), 0, sizeof(wr.wr));
  ibv_send_wr* bad_wr = nullptr;
  PCHECK(ibv_post_send(qp_, &wr, &bad_wr) == 0);
}

void IBVerbsQP::ReadDone(WorkRequestId* wr_id) {
  CHECK_GE(wr_id->outstanding_sge_cnt, 1);
  wr_id->outstanding_sge_cnt -= 1;
  if (wr_id->outstanding_sge_cnt == 0) {
    Global<CommNet>::Get()->ReadDone(wr_id->stream_id);
    DeleteWorkRequestId(wr_id);
  }
}

void IBVerbsQP::SendDone(WorkRequestId* wr_id) {
  {
    std::unique_lock<std::mutex> lck(send_msg_buf_mtx_);
    send_msg_buf_.push(wr_id->msg_mr);
  }
  DeleteWorkRequestId(wr_id);
}

void IBVerbsQP::RecvDone(WorkRequestId* wr_id) {
  Global<ActorMsgBus>::Get()->SendMsgWithoutCommNet(wr_id->msg_mr->msg());
  PostRecvRequest(wr_id->msg_mr);
  DeleteWorkRequestId(wr_id);
}

void IBVerbsQP::PostRecvRequest(ActorMsgMR* msg_mr) {
  WorkRequestId* wr_id = NewWorkRequestId();
  wr_id->msg_mr = msg_mr;
  ibv_recv_wr wr;
  wr.wr_id = reinterpret_cast<uint64_t>(wr_id);
  wr.next = nullptr;
  wr.sg_list = const_cast<ibv_sge*>(&(msg_mr->mem_desc().sge_vec().at(0)));
  wr.num_sge = 1;
  ibv_recv_wr* bad_wr = nullptr;
  PCHECK(ibv_post_recv(qp_, &wr, &bad_wr) == 0);
}

ActorMsgMR* IBVerbsQP::GetOneSendMsgMRFromBuf() {
  std::unique_lock<std::mutex> lck(send_msg_buf_mtx_);
  if (send_msg_buf_.empty()) { send_msg_buf_.push(new ActorMsgMR(pd_)); }
  ActorMsgMR* msg_mr = send_msg_buf_.front();
  send_msg_buf_.pop();
  return msg_mr;
}

WorkRequestId* IBVerbsQP::NewWorkRequestId() {
  WorkRequestId* wr_id = new WorkRequestId;
  wr_id->qp = this;
  wr_id->outstanding_sge_cnt = 0;
  wr_id->stream_id = -1;
  wr_id->msg_mr = nullptr;
  return wr_id;
}

void IBVerbsQP::DeleteWorkRequestId(WorkRequestId* wr_id) {
  CHECK_EQ(wr_id->qp, this);
  delete wr_id;
}

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX
