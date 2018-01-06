#ifdef WITH_RDMA

#include "oneflow/core/comm_network/rdma/endpoint_manager.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/actor/actor_message_bus.h"

namespace oneflow {

namespace {

std::string GenConnInfoKey(int64_t src_machine_id, int64_t dst_machine_id) {
  return "RdmaConnInfo/" + std::to_string(src_machine_id) + " "
         + std::to_string(dst_machine_id);
}

}  // namespace

EndpointManager::EndpointManager() {
  // Init Adapter
  ibv_device** device_list = ibv_get_device_list(NULL);
  ibv_device* device = device_list[0];
  context_ = ibv_open_device(device);
  CHECK(context_);
  pd_ = ibv_alloc_pd(context_);
  CHECK(pd_);

  // Init env
  send_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe
  CHECK(send_cq_);
  recv_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe
  CHECK(recv_cq_);

  ibv_free_device_list(device_list);
  LOG(INFO) << "Finish EndpointManager Constructor";
}

EndpointManager::~EndpointManager() {
  for (auto it = send_msg2rdma_mem_.begin(); it != send_msg2rdma_mem_.end();
       ++it) {
    delete it->first;
    delete it->second;
  }
  for (auto it = recv_msg2rdma_mem_.begin(); it != recv_msg2rdma_mem_.end();
       ++it) {
    delete it->first;
    delete it->second;
  }
  for (auto it = connection_pool_.begin(); it != connection_pool_.end(); ++it) {
    delete it->second;
  }
  if (send_cq_ != nullptr) { CHECK_EQ(ibv_destroy_cq(send_cq_), 0); }
  if (recv_cq_ != nullptr) { CHECK_EQ(ibv_destroy_cq(recv_cq_), 0); }
  if (pd_ != nullptr) { CHECK_EQ(ibv_dealloc_pd(pd_), 0); }
  if (context_ != nullptr) { CHECK_EQ(ibv_close_device(context_), 0); }
}

void EndpointManager::InitRdma() {
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  int64_t this_machine_id = MachineCtx::Singleton()->this_machine_id();
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    if (peer_machine_id == this_machine_id) { continue; }
    Connection* conn = NewConnection();
    connection_pool_.emplace(peer_machine_id, conn);
    CtrlClient::Singleton()->PushKV(
        GenConnInfoKey(this_machine_id, peer_machine_id),
        conn->mut_this_machine_conn_info());
  }
  OF_BARRIER();
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    if (peer_machine_id == this_machine_id) { continue; }
    Connection* conn = connection_pool_[peer_machine_id];
    CtrlClient::Singleton()->PullKV(
        GenConnInfoKey(peer_machine_id, this_machine_id),
        conn->mut_peer_machine_conn_info_ptr());
    LOG(INFO) << "Connection " << reinterpret_cast<uint64_t>(conn)
              << " info: " << conn->mut_peer_machine_conn_info().lid() << " "
              << conn->mut_peer_machine_conn_info().qpn() << " "
              << conn->mut_peer_machine_conn_info().psn() << " "
              << conn->mut_peer_machine_conn_info().snp() << " "
              << conn->mut_peer_machine_conn_info().iid();
    for (size_t i = 0; i != kPrePostRecvNum; ++i) {
      ActorMsg* actor_msg = new ActorMsg();
      const RdmaMem* rdma_mem = NewRdmaMem(actor_msg, sizeof(ActorMsg));
      recv_msg2conn_ptr_.emplace(actor_msg, conn);
      recv_msg2rdma_mem_.emplace(actor_msg, rdma_mem);
      conn->PostRecvRequest(actor_msg, rdma_mem);
    }
    conn->CompleteConnection();
  }
  OF_BARRIER();
  LOG(INFO) << "Finish InitRdma";
}

RdmaMem* EndpointManager::NewRdmaMem(void* mem_ptr, size_t byte_size) {
  return new RdmaMem(pd_, mem_ptr, byte_size);
}

Connection* EndpointManager::NewConnection() {
  Connection* conn = new Connection();

  // Init queue pair
  ibv_qp_init_attr qp_init_attr;

  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.qp_context = nullptr;
  qp_init_attr.send_cq = send_cq_;
  qp_init_attr.recv_cq = recv_cq_;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.srq = nullptr;
  qp_init_attr.sq_sig_all = 1;

  qp_init_attr.cap.max_send_wr = 10;
  qp_init_attr.cap.max_recv_wr = 10;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;

  ibv_qp* qp_ptr = ibv_create_qp(pd_, &qp_init_attr);
  CHECK(qp_ptr);

  // Init connection info
  ibv_port_attr attr;
  CHECK_EQ(ibv_query_port(context_, (uint8_t)1, &attr), 0);
  srand((unsigned)time(NULL));
  conn->mut_this_machine_conn_info_ptr()->set_lid(attr.lid);
  conn->mut_this_machine_conn_info_ptr()->set_qpn(qp_ptr->qp_num);
  conn->mut_this_machine_conn_info_ptr()->set_psn(static_cast<uint32_t>(rand())
                                                  & 0xffffff);
  union ibv_gid gid;
  CHECK_EQ(ibv_query_gid(context_, (uint8_t)1, 0, &gid), 0);
  conn->mut_this_machine_conn_info_ptr()->set_snp(gid.global.subnet_prefix);
  conn->mut_this_machine_conn_info_ptr()->set_iid(gid.global.interface_id);
  conn->set_ibv_mtu(attr.active_mtu);
  conn->set_ibv_qp_ptr(qp_ptr);
  LOG(INFO) << "Connection " << reinterpret_cast<uint64_t>(conn)
            << " info: " << conn->mut_this_machine_conn_info().lid() << " "
            << conn->mut_this_machine_conn_info().qpn() << " "
            << conn->mut_this_machine_conn_info().psn() << " "
            << conn->mut_this_machine_conn_info().snp() << " "
            << conn->mut_this_machine_conn_info().iid();
  return conn;
}

void EndpointManager::Read(void* read_ctx, int64_t src_machine_id,
                           const RdmaMem* local_mem,
                           const RdmaMemDesc& remote_mem_desc) {
  auto iter = connection_pool_.find(src_machine_id);
  CHECK(iter != connection_pool_.end());
  Connection* conn = iter->second;
  conn->PostReadRequest(read_ctx, local_mem, remote_mem_desc);
}

void EndpointManager::SendActorMsg(int64_t dst_machine_id,
                                   const ActorMsg& msg) {
  auto iter = connection_pool_.find(dst_machine_id);
  CHECK(iter != connection_pool_.end());
  Connection* conn = iter->second;
  std::tuple<ActorMsg*, RdmaMem*> allocate_ret = AllocateSendMsg();
  ActorMsg* msg_ptr = std::get<0>(allocate_ret);
  *msg_ptr = msg;
  conn->PostSendRequest(msg_ptr, std::get<1>(allocate_ret));
}

void EndpointManager::Start() {
  thread_state_ = true;
  thread_ = std::thread(&EndpointManager::PollLoop, this);
}

void EndpointManager::Stop() {
  thread_state_ = false;
  thread_.join();
}

void EndpointManager::PollLoop() {
  while (true) {
    if (!thread_state_) { return; }
    PollSendQueue();
    PollRecvQueue();
  }
}

void EndpointManager::PollSendQueue() {
  ibv_wc wc;
  int32_t len = ibv_poll_cq(send_cq_, 1, &wc);

  if (len <= 0) { return; }

  if (wc.status != IBV_WC_SUCCESS) {
    LOG(INFO) << "PollSend wc.status != WC_SUCCESS " << wc.status;
    return;
  }
  switch (wc.opcode) {
    case IBV_WC_SEND: {
      ReleaseSendMsg(reinterpret_cast<ActorMsg*>(wc.wr_id));
      return;
    }
    case IBV_WC_RDMA_READ: {
      CommNet::Singleton()->ReadDone(reinterpret_cast<void*>(wc.wr_id));
      return;
    }
    default: return;
  }
}

void EndpointManager::PollRecvQueue() {
  ibv_wc wc;
  int32_t len = ibv_poll_cq(recv_cq_, 1, &wc);

  if (len <= 0) { return; }

  if (wc.status != IBV_WC_SUCCESS) {
    LOG(INFO) << "PollRecv wc.status != WC_SUCCESS " << wc.status;
    return;
  }
  const ActorMsg* msg = reinterpret_cast<const ActorMsg*>(wc.wr_id);

  ActorMsgBus::Singleton()->SendMsg(*msg);
  CHECK(recv_msg2conn_ptr_.find(msg) != recv_msg2conn_ptr_.end());
  Connection* conn = recv_msg2conn_ptr_.at(msg);
  auto msg2mem_it = recv_msg2rdma_mem_.find(msg);
  conn->PostRecvRequest(msg, msg2mem_it->second);
}

std::tuple<ActorMsg*, RdmaMem*> EndpointManager::AllocateSendMsg() {
  std::unique_lock<std::mutex> lck(send_msg_pool_mutex_);
  if (send_msg_pool_.empty()) {
    ActorMsg* msg = new ActorMsg;
    RdmaMem* mem = NewRdmaMem(msg, sizeof(ActorMsg));
    send_msg2rdma_mem_.emplace(msg, mem);
    send_msg_pool_.push(msg);
  }
  ActorMsg* ret_msg = send_msg_pool_.front();
  send_msg_pool_.pop();
  auto send_msg2rdma_mem_it = send_msg2rdma_mem_.find(ret_msg);
  CHECK(send_msg2rdma_mem_it != send_msg2rdma_mem_.end());
  return std::make_tuple(ret_msg, send_msg2rdma_mem_it->second);
}

void EndpointManager::ReleaseSendMsg(ActorMsg* msg) {
  std::unique_lock<std::mutex> lck(send_msg_pool_mutex_);
  send_msg_pool_.push(msg);
}

}  // namespace oneflow

#endif  // WITH_RDMA
