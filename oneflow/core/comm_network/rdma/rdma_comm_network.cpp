#ifdef WITH_RDMA

#include "oneflow/core/comm_network/rdma/rdma_comm_network.h"
#include "oneflow/core/comm_network/rdma/rdma_tokens_message.pb.h"

namespace oneflow {

namespace {

std::string GenTokensMsgKey(int64_t machine_id) {
  return "RdmaTokensMsg/" + std::to_string(machine_id);
}

}  // namespace

void RdmaCommNet::Init() {
  CommNet::Singleton()->set_comm_network_ptr(new RdmaCommNet());
}

RdmaCommNet::RdmaCommNet() {
  mems_.clear();
  unregister_mems_cnt_ = 0;
  endpoint_manager_ = new EndpointManager();
  endpoint_manager_->InitRdma();
  endpoint_manager_->Start();
}

RdmaCommNet::~RdmaCommNet() {
  endpoint_manager_->Stop();
  delete endpoint_manager_;
  CHECK(mems_.empty());
}

const void* RdmaCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  RdmaMem* rdma_mem = endpoint_manager_->NewRdmaMem(mem_ptr, byte_size);
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
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  int64_t this_machine_id = MachineCtx::Singleton()->this_machine_id();
  RdmaTokensMsg this_machine_tokens_msg;
  for (RdmaMem* mem_ptr : mems_) {
    this_machine_tokens_msg.mutable_token2mem_desc()->insert(
        {reinterpret_cast<uint64_t>(mem_ptr), mem_ptr->GenRdmaMemDesc()});
  }
  CtrlClient::Singleton()->PushKV(GenTokensMsgKey(this_machine_id),
                                  this_machine_tokens_msg);
  OF_BARRIER();
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    if (peer_machine_id == MachineCtx::Singleton()->this_machine_id()) {
      continue;
    }
    RdmaTokensMsg peer_machine_tokens_msg;
    CtrlClient::Singleton()->PullKV(GenTokensMsgKey(peer_machine_id),
                                    &peer_machine_tokens_msg);
    HashMap<uint64_t, RdmaMemDesc> peer_token2mem_desc =
        PbMap2HashMap(peer_machine_tokens_msg.token2mem_desc());
    for (auto pair : peer_token2mem_desc) {
      token2mem_desc_.insert({pair.first, pair.second});
    }
  }
  OF_BARRIER();
  LOG(INFO) << "Finish RegisterMemoryDone";
}

void* RdmaCommNet::Read(void* actor_read_id, int64_t src_machine_id,
                        const void* src_token, const void* dst_token) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = new ReadContext;
  read_ctx->done_cnt = 0;
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    actor_read_ctx->read_ctx_list.push_back(read_ctx);
  }
  RdmaMemDesc& remote_mem_desc =
      token2mem_desc_[reinterpret_cast<uint64_t>(src_token)];
  auto local_mem = static_cast<const RdmaMem*>(dst_token);
  void* read_done_id =
      new std::tuple<ActorReadContext*, ReadContext*>(actor_read_ctx, read_ctx);
  endpoint_manager_->Read(read_done_id, src_machine_id, local_mem,
                          remote_mem_desc);
  return read_ctx;
}

void RdmaCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  endpoint_manager_->SendActorMsg(dst_machine_id, msg);
}

}  // namespace oneflow

#endif  // WITH_RDMA
