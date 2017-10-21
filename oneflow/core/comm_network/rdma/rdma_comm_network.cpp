#include "oneflow/core/comm_network/rdma/rdma_comm_network.h"

const void* RdmaCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  auto rdma_mem = new RdmaMem;
  rdma_mem->Register(mem_ptr, byte_size);
  {
    std::unique_lock<std::mutex> lck(mem_mutex_);
    mems_.push_back(rdma_mem);
  }
  returen rdma_mem;
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

void* RdmaCommNet::Read(void* actor_read_id, int64_t src_machine_id,
    const void* src_token, const void* dst_token) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = new ReadContext;;
  read_ctx->done_cnt = 0;
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    actor_read_ctx->read_ctx_list.push_back(read_ctx);
  }
  auto remote_mem_desc = static_cast<RdmaMemDesc*>(src_token)
  auto local_mem = static_cast<RdmaMem*>(dst_token);
  auto conn = connection_pool_->GetConnection(src_machine_id);
  conn->PostReadRequest(read_ctx, local_mem, remote_mem_desc); 
  return read_ctx;
}

void RdmaCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  auto rdma_mem = RegisterMemory(&msg, size_of(msg));
  auto conn = connection_pool->GetConnection(dst_machine_id);
  conn->PostSendRequest(rdma_mem);
}

