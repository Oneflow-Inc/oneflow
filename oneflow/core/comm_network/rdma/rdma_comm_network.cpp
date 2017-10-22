#include "oneflow/core/comm_network/rdma/rdma_comm_network.h"

void RdmaCommNet::Init() {
  CommNet::Singleton()->set_comm_network_ptr(new RdmaCommNet());
}

RdmaCommNet::RdmaCommNet() {
  mems_.clean();
  unregister_mems_cnt_ = 0;
  InitRdma();
}

Connection* RdmaCommNet::NewConnection() {
  Connection* conn = new Connection();
  // TODO
}

RdmaCommNet::InitRdma() {
  int64_t this_machine_id = RuntimeCtx::Singleton()->this_machine_id();
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  CtrlClient::Singleton()->PushConnectorInfo();
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    if (peer_machine_id == this_machine_id) continue;
    ConnectorInfo& peer_conn_info =
        CtrlClient::Singleton()->PullConnectorInfo(peer_machine_id);
    Connection* conn = NewConnection();
    conn.set_peer_conn_info(peer_conn_info);
    connection_pool_.AddConnection(peer_machine_id, conn);
  }
  CtrlClient::Singleton()->Barrier();
  CtrlClient::Singleton()->CleanConnectionInfo(this_machine_id);
}

const void* RdmaCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  RdmaMem* rdma_mem = endpoint_manager_.NewRdmaMem();
  rdma_mem->Register(mem_ptr, byte_size);
  {
    std::unique_lock<std::mutex> lck(mem_mutex_);
    mems_.push_back(rdma_mem);
  }
  return rdma_mem;
}

void RdmaCommNet::UnRegisterMemory(const void* token) {
  std::unique_lock<std::mutex> lck(mem_mutex_);
  CHECK(!mems_.empty());
  unregistered_mems_cnt_ += 1;
  if (unregister_mems_cnt_ == mems_.size()) {
    for (RdmaMem* mem : mems_) { delete mem; }
    mems_.clear();
    unregister_mems_cnt_ = 0;
  }
}

void RdmaCommNet::RegisterMemoryDone() {}

void* RdmaCommNet::Read(void* actor_read_id, int64_t src_machine_id,
                        const void* src_token, const void* dst_token) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = new ReadContext;
  ;
  read_ctx->done_cnt = 0;
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    actor_read_ctx->read_ctx_list.push_back(read_ctx);
  }
  auto remote_mem_desc = static_cast<RdmaMemDesc*>(src_token);
  auto local_mem = static_cast<RdmaMem*>(dst_token);
  auto conn = connection_pool_->GetConnection(src_machine_id);
  conn->PostReadRequest(read_ctx, local_mem, remote_mem_desc);
  return read_ctx;
}

void RdmaCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  auto rdma_mem = RegisterMemory(&msg, sizeof(msg));
  auto conn = connection_pool_->GetConnection(dst_machine_id);
  conn->PostSendRequest(rdma_mem);
}
