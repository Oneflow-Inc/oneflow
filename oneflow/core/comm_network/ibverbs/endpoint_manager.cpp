#include "oneflow/core/comm_network/ibverbs/endpoint_manager.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/actor/actor_message_bus.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

namespace oneflow {

namespace {

std::string GenConnInfoKey(int64_t src_machine_id, int64_t dst_machine_id) {
  return "IBVerbsConnInfo/" + std::to_string(src_machine_id) + " "
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
  InitRdma();
  Start();
}

EndpointManager::~EndpointManager() {
  Stop();
  for (auto& pair : send_msg2mem_desc_) {
    delete pair.first;
    delete pair.second;
  }
  for (auto& pair : recv_msg2mem_desc_) {
    delete pair.first;
    delete pair.second;
  }
  for (auto& pair : connection_pool_) { delete pair.second; }
  if (send_cq_ != nullptr) { CHECK_EQ(ibv_destroy_cq(send_cq_), 0); }
  if (recv_cq_ != nullptr) { CHECK_EQ(ibv_destroy_cq(recv_cq_), 0); }
  if (pd_ != nullptr) { CHECK_EQ(ibv_dealloc_pd(pd_), 0); }
  if (context_ != nullptr) { CHECK_EQ(ibv_close_device(context_), 0); }
}

void EndpointManager::InitRdma() {
  int64_t this_machine_id = MachineCtx::Singleton()->this_machine_id();
  for (int64_t peer_machine_id : CommNet::Singleton()->peer_machine_id()) {
    IBVerbsConnection* conn = NewIBVerbsConnection();
    connection_pool_.emplace(peer_machine_id, conn);
    CtrlClient::Singleton()->PushKV(
        GenConnInfoKey(this_machine_id, peer_machine_id),
        conn->mut_this_machine_conn_info());
  }
  OF_BARRIER();
  for (int64_t peer_machine_id : CommNet::Singleton()->peer_machine_id()) {
    IBVerbsConnection* conn = connection_pool_[peer_machine_id];
    CtrlClient::Singleton()->PullKV(
        GenConnInfoKey(peer_machine_id, this_machine_id),
        conn->mut_peer_machine_conn_info_ptr());
    for (size_t i = 0; i != kPrePostRecvNum; ++i) {
      ActorMsg* actor_msg = new ActorMsg;
      auto ibverbs_mem_desc = NewIBVerbsMemDesc(actor_msg, sizeof(ActorMsg));
      recv_msg2conn_ptr_.emplace(actor_msg, conn);
      recv_msg2mem_desc_.emplace(actor_msg, ibverbs_mem_desc);
      conn->PostRecvRequest(actor_msg, ibverbs_mem_desc);
    }
    conn->CompleteConnection();
  }
  OF_BARRIER();
}

IBVerbsMemDesc* EndpointManager::NewIBVerbsMemDesc(void* mem_ptr,
                                                   size_t byte_size) {
  return new IBVerbsMemDesc(pd_, mem_ptr, byte_size);
}

IBVerbsConnection* EndpointManager::NewIBVerbsConnection() {
  IBVerbsConnection* conn = new IBVerbsConnection();

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
  return conn;
}

void EndpointManager::Read(void* read_ctx, int64_t src_machine_id,
                           IBVerbsMemDesc* local_mem_desc,
                           IBVerbsMemDescProto& remote_mem_desc_proto) {
  auto iter = connection_pool_.find(src_machine_id);
  CHECK(iter != connection_pool_.end());
  IBVerbsConnection* conn = iter->second;
  conn->PostReadRequest(read_ctx, local_mem_desc, remote_mem_desc_proto);
}

void EndpointManager::SendActorMsg(int64_t dst_machine_id,
                                   const ActorMsg& msg) {
  auto iter = connection_pool_.find(dst_machine_id);
  CHECK(iter != connection_pool_.end());
  IBVerbsConnection* conn = iter->second;
  std::tuple<ActorMsg*, IBVerbsMemDesc*> allocate_ret = AllocateSendMsg();
  ActorMsg* msg_ptr = std::get<0>(allocate_ret);
  *msg_ptr = msg;
  conn->PostSendRequest(msg_ptr, std::get<1>(allocate_ret));
}

void EndpointManager::Start() {
  poll_state_ = true;
  poll_thread_ = std::thread(&EndpointManager::PollLoop, this);
}

void EndpointManager::Stop() {
  poll_state_ = false;
  poll_thread_.join();
}

void EndpointManager::PollLoop() {
  while (true) {
    if (!poll_state_) { return; }
    PollSendQueue();
    PollRecvQueue();
  }
}

void EndpointManager::PollSendQueue() {
  ibv_wc wc;
  int32_t len = ibv_poll_cq(send_cq_, 1, &wc);

  if (len <= 0) { return; }

  if (wc.status != IBV_WC_SUCCESS) {
    LOG(FATAL) << "PollSendQueue Error Code: " << wc.status;
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
    LOG(FATAL) << "PollRecvQueue Error Code:  " << wc.status;
  }
  ActorMsg* msg = reinterpret_cast<ActorMsg*>(wc.wr_id);

  ActorMsgBus::Singleton()->SendMsg(*msg);
  CHECK(recv_msg2conn_ptr_.find(msg) != recv_msg2conn_ptr_.end());
  IBVerbsConnection* conn = recv_msg2conn_ptr_.at(msg);
  auto msg2mem_it = recv_msg2mem_desc_.find(msg);
  conn->PostRecvRequest(msg, msg2mem_it->second);
}

std::tuple<ActorMsg*, IBVerbsMemDesc*> EndpointManager::AllocateSendMsg() {
  std::unique_lock<std::mutex> lck(send_msg_pool_mutex_);
  if (send_msg_pool_.empty()) {
    ActorMsg* msg = new ActorMsg;
    IBVerbsMemDesc* mem_desc = NewIBVerbsMemDesc(msg, sizeof(ActorMsg));
    send_msg2mem_desc_.emplace(msg, mem_desc);
    send_msg_pool_.push(msg);
  }
  ActorMsg* ret_msg = send_msg_pool_.front();
  send_msg_pool_.pop();
  auto send_msg2mem_desc_it = send_msg2mem_desc_.find(ret_msg);
  CHECK(send_msg2mem_desc_it != send_msg2mem_desc_.end());
  return std::make_tuple(ret_msg, send_msg2mem_desc_it->second);
}

void EndpointManager::ReleaseSendMsg(ActorMsg* msg) {
  std::unique_lock<std::mutex> lck(send_msg_pool_mutex_);
  send_msg_pool_.push(msg);
}

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX
