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

RdmaCommNet::~RdmaCommNet() {
  // TODO
}

Connection* RdmaCommNet::NewConnection() {
  Connection* conn = endpoint_manager_->NewConnection();
  return conn;
}

ConnectionInfo& RdmaCommNet::GetMachineConnInfo() {
  return endpoint_manager_->GetMachineConnInfo();
}

void RdmaCommNet::InitRdma() {
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  CtrlClient::Singleton()->PushConnectionInfo(GetMachineConnInfo());
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    Connection* conn = NewConnection();
    conn->set_peer_conn_info(
        CtrlClient::Singleton()->PullConnectionInfo(peer_machine_id));
    connection_pool_->AddConnection(peer_machine_id, conn);
  }
  OF_BARRIER();
  CtrlClient::Singleton()->ClearConnectionInfo();
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

void RdmaCommNet::RegisterMemoryDone() {
  // TODO
}

void* RdmaCommNet::NewActorReadId() {
  // TODO
  return nullptr;
}

void RdmaCommNet::DeleteActorReadId(void* actor_read_id) {
  // TODO
}

void RdmaCommNet::AddReadCallBack(void* actor_read_id, void* read_id,
                                  std::function<void()> callback) {
  // TODO
}

void RdmaCommNet::AddReadCallBackDone(void* actor_read_id, void* read_id) {
  // TODO
}

void RdmaCommNet::ReadDone(void* read_done_id) {
  // TODO
}

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
  // TODO
  RdmaMemDesc* remote_mem_desc = nullptr;
  auto local_mem = static_cast<const RdmaMem*>(dst_token);
  endpoint_manager_->Read(read_ctx, src_machine_id, local_mem, remote_mem_desc);
  return read_ctx;
}

void RdmaCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  // auto rdma_mem = static_cast<const RdmaMem*>(RegisterMemory(
  //    reinterpret_cast<void*>(const_cast<ActorMsg*>(&msg)), sizeof(msg)));
  endpoint_manager_->SendActorMsg(dst_machine_id, msg);
}

int8_t RdmaCommNet::IncreaseDoneCnt(ReadContext* read_ctx) {
  // TODO
  return 0;
}

void RdmaCommNet::FinishOneReadContext(ActorReadContext* actor_read_ctx,
                                       ReadContext* read_ctx) {
  // TODO
}

}  // namespace oneflow
