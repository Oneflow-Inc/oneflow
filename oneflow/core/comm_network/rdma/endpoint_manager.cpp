#include "oneflow/core/comm_network/rdma/endpoint_manager.h"
#include <arpa/inet.h>
#include "glog/logging.h"
#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

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
  LOG(INFO) << "EndpointManager Constructed!";
}

ibv_qp* EndpointManager::CreateQueuePair() {
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

  ibv_qp* qp_ptr_ = ibv_create_qp(pd_, &qp_init_attr);
  CHECK(qp_ptr_);
  return qp_ptr_;
}

EndpointManager::~EndpointManager() {
  for (auto it = recv_msg2rdma_mem_.begin(); it != recv_msg2rdma_mem_.end();
       ++it) {
    delete it->first;
    CommNet::Singleton()->UnRegisterMemory(it->second);
  }
  if (send_cq_ != nullptr) { CHECK_EQ(ibv_destroy_cq(send_cq_), 0); }
  if (recv_cq_ != nullptr) { CHECK_EQ(ibv_destroy_cq(recv_cq_), 0); }
  if (pd_ != nullptr) { CHECK_EQ(ibv_dealloc_pd(pd_), 0); }
  if (context_ != nullptr) { CHECK_EQ(ibv_close_device(context_), 0); }
}

void EndpointManager::InitRdma() {
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  CtrlClient::Singleton()->PushConnectionInfo(GetMachineConnInfo());
  // TODO this_mach_conn_info no difference for each connection
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    if (peer_machine_id == RuntimeCtx::Singleton()->this_machine_id()) {
      continue;
    }
    Connection* conn = NewConnection();
    LOG(INFO) << "Before PullConnectionInfo";
    conn->mut_peer_conn_info() =
        CtrlClient::Singleton()->PullConnectionInfo(peer_machine_id);
    LOG(INFO) << "After PullConnectionInfo";
    connection_pool_.emplace(peer_machine_id, conn);
    for (size_t i = 0; i != kPrePostRecvNum; ++i) {
      ActorMsg* actor_msg = new ActorMsg();
      const RdmaMem* rdma_mem = static_cast<const RdmaMem*>(
          CommNet::Singleton()->RegisterMemory(actor_msg, sizeof(ActorMsg)));
      recv_msg2rdma_mem_.emplace(actor_msg, const_cast<RdmaMem*>(rdma_mem));
      conn->PostRecvRequest(actor_msg, rdma_mem);
    }
    conn->CompleteConnection();
  }
  LOG(INFO) << "Before barrier";
  OF_BARRIER();
  LOG(INFO) << "After barrier";
  CtrlClient::Singleton()->ClearConnectionInfo();
  LOG(INFO) << "InitRdma finished!";
}

RdmaMem* EndpointManager::NewRdmaMem() {
  RdmaMem* rdma_mem = new RdmaMem(pd_);
  CHECK(rdma_mem);
  return rdma_mem;
}

Connection* EndpointManager::NewConnection() {
  Connection* conn = new Connection();
  // Init connection info
  ibv_port_attr attr;
  CHECK_EQ(ibv_query_port(context_, (uint8_t)1, &attr), 0);
  srand((unsigned)time(NULL));
  conn->mut_this_mach_conn_info().set_lid(attr.lid);
  // Will be set up after the creation of the queue pair
  conn->mut_this_mach_conn_info().set_qpn(0);
  conn->mut_this_mach_conn_info().set_psn(static_cast<uint32_t>(rand())
                                          & 0xffffff);
  union ibv_gid gid;
  CHECK_EQ(ibv_query_gid(context_, (uint8_t)1, 0, &gid), 0);
  conn->mut_this_mach_conn_info().set_snp(gid.global.subnet_prefix);
  conn->mut_this_mach_conn_info().set_iid(gid.global.interface_id);
  conn->set_ibv_mtu(attr.active_mtu);
  conn->set_ibv_qp_ptr(CreateQueuePair());
  return conn;
}

void EndpointManager::Read(void* read_ctx, int64_t src_machine_id,
                           const RdmaMem* local_mem,
                           const RdmaMemDesc& remote_mem_desc) {
  Connection* conn = connection_pool_[src_machine_id];
  conn->PostReadRequest(read_ctx, local_mem, remote_mem_desc);
  LOG(INFO) << "Read";
}

void EndpointManager::SendActorMsg(int64_t dst_machine_id,
                                   const ActorMsg& msg) {
  Connection* conn = connection_pool_[dst_machine_id];
  ActorMsg* msg_ptr = new ActorMsg;
  *msg_ptr = msg;
  const void* rdma_mem =
      CommNet::Singleton()->RegisterMemory(msg_ptr, sizeof(*msg_ptr));
  conn->PostSendRequest(msg_ptr, static_cast<const RdmaMem*>(rdma_mem));
}

void EndpointManager::Start() {
  thread_state_ = true;
  thread_ = std::thread(&EndpointManager::PollLoop, this);
  LOG(INFO) << "PollLoop start!";
}

void EndpointManager::Stop() {
  thread_state_ = false;
  thread_.join();
}

void EndpointManager::PollLoop() {
  LOG(INFO) << "Enter PollLoop";
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

  if (wc.status != IBV_WC_SUCCESS) { return; }
  LOG(INFO) << "Poll Send Queue";
  switch (wc.opcode) {
    case IBV_WC_SEND: {
      CommNet::Singleton()->UnRegisterMemory(
          reinterpret_cast<const void*>(wc.wr_id));
      delete reinterpret_cast<ActorMsg*>(wc.wr_id);
      LOG(INFO) << "Successfully send ActorMsg";
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

  if (wc.status != IBV_WC_SUCCESS) { return; }
  ActorMsg* msg = reinterpret_cast<ActorMsg*>(wc.wr_id);

  ActorMsgBus::Singleton()->SendMsg(*msg);
  LOG(INFO) << "Successfully recv ActorMsg";
  int64_t src_actor_id = msg->src_actor_id();
  Connection* conn = connection_pool_[src_actor_id];
  auto msg2mem_it = recv_msg2rdma_mem_.find(msg);
  conn->PostRecvRequest(msg, msg2mem_it->second);
}

}  // namespace oneflow
