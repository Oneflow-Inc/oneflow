#include "oneflow/core/comm_network/rdma/rdma_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"

namespace oneflow {

void RdmaCommNet::Init() {
  CommNet::Singleton()->set_comm_network_ptr(new RdmaCommNet());
}

RdmaCommNet::RdmaCommNet() {
  mems_.clear();
  unregister_mems_cnt_ = 0;
  InitRdma();
}

Connection* RdmaCommNet::NewConnection() {
  Connection* conn = new Connection();
  // TODO
  return conn;
}

void RdmaCommNet::InitRdma() {
  int64_t this_machine_id = RuntimeCtx::Singleton()->this_machine_id();
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  FOR_RANGE(int64_t, peer_machine_id, this_machine_id + 1, total_machine_num) {
    Connection* conn = NewConnection();
    conn->ConnectTo(peer_machine_id);
    connection_pool_->AddConnection(peer_machine_id, conn);
  }
  CtrlClient::Singleton()->PushConnectorInfo();
  FOR_RANGE(int64_t, idx, 0, this_machine_id) {
    Connection* conn = NewConnection();
    conn->WaitForConnection();
    connection_pool_->AddConnection(idx, conn);
  }
}

const void* RdmaCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  RdmaMem* rdma_mem = endpoint_manager_->NewRdmaMem();
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
  unregister_mems_cnt_ += 1;
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
  auto remote_mem_desc = static_cast<const RdmaMemDesc*>(src_token);
  auto local_mem = static_cast<const RdmaMem*>(dst_token);
  auto conn = connection_pool_->GetConnection(src_machine_id);
  conn->PostReadRequest(read_ctx, local_mem, remote_mem_desc);
  return read_ctx;
}

void RdmaCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  auto rdma_mem = static_cast<const RdmaMem*>(RegisterMemory(
      reinterpret_cast<void*>(const_cast<ActorMsg*>(&msg)), sizeof(msg)));
  auto conn = connection_pool_->GetConnection(dst_machine_id);
  conn->PostSendRequest(rdma_mem);
}

}  // namespace oneflow
