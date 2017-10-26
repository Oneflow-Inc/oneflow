#include "oneflow/core/comm_network/rdma/rdma_comm_network.h"

namespace oneflow {

void RdmaCommNet::Init() {
  CommNet::Singleton()->set_comm_network_ptr(new RdmaCommNet());
}

RdmaCommNet::RdmaCommNet() {
  mems_.clear();
  unregister_mems_cnt_ = 0;
  rdma_established_ = false;
  endpoint_manager_.reset(new EndpointManager());
  endpoint_manager_->Start();
}

RdmaCommNet::~RdmaCommNet() {
  endpoint_manager_->Stop();
  CHECK(mems_.empty());
}

const void* RdmaCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  RdmaMem* rdma_mem = endpoint_manager_->NewRdmaMem();
  rdma_mem->Register(mem_ptr, byte_size);
  {
    std::unique_lock<std::mutex> lck(mem_mutex_);
    if (!rdma_established_) {
      endpoint_manager_->InitRdma();
      rdma_established_ = true;
    }
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
  HashMap<uint64_t, RdmaMemDesc> this_machine_token_msgs;
  for (RdmaMem* mem_ptr : mems_) {
    this_machine_token_msgs.insert(
        {reinterpret_cast<uint64_t>(mem_ptr), mem_ptr->GetRdmaMemDesc()});
  }
  TokenMsgs token_msgs;
  *(token_msgs.mutable_token2mem_desc()) =
      HashMap2PbMap<uint64_t, RdmaMemDesc>(this_machine_token_msgs);
  CtrlClient::Singleton()->PushTokenMsgs(token_msgs);
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    auto& peer_token_msgs =
        CtrlClient::Singleton()->PullTokenMsgs(peer_machine_id);
    for (auto mem_desc_pair : peer_token_msgs.token2mem_desc()) {
      token2mem_desc_.insert({mem_desc_pair.first, mem_desc_pair.second});
    }
  }
  OF_BARRIER();
  CtrlClient::Singleton()->ClearTokenMsgs();
}

void* RdmaCommNet::NewActorReadId() { return new ActorReadContext; }

void RdmaCommNet::DeleteActorReadId(void* actor_read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->read_ctx_list.empty());
  delete actor_read_ctx;
}

void RdmaCommNet::AddReadCallBack(void* actor_read_id, void* read_id,
                                  std::function<void()> callback) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if (read_ctx) {
    read_ctx->cbl.push_back(callback);
    return;
  }
  do {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    if (actor_read_ctx->read_ctx_list.empty()) {
      break;
    } else {
      actor_read_ctx->read_ctx_list.back()->cbl.push_back(callback);
      return;
    }
  } while (0);
  callback();
}

void RdmaCommNet::AddReadCallBackDone(void* actor_read_id, void* read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if (IncreaseDoneCnt(read_ctx) == 2) {
    FinishOneReadContext(actor_read_ctx, read_ctx);
    delete read_ctx;
  }
}

void RdmaCommNet::ReadDone(void* read_done_id) {
  auto parsed_read_done_id =
      static_cast<std::tuple<ActorReadContext*, ReadContext*>*>(read_done_id);
  auto actor_read_ctx = std::get<0>(*parsed_read_done_id);
  auto read_ctx = std::get<1>(*parsed_read_done_id);
  delete parsed_read_done_id;
  if (IncreaseDoneCnt(read_ctx) == 2) {
    {
      std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
      FinishOneReadContext(actor_read_ctx, read_ctx);
    }
    delete read_ctx;
  }
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
  RdmaMemDesc& remote_mem_desc =
      token2mem_desc_[reinterpret_cast<int64_t>(src_token)];
  auto local_mem = static_cast<const RdmaMem*>(dst_token);
  endpoint_manager_->Read(read_ctx, src_machine_id, local_mem, remote_mem_desc);
  return read_ctx;
}

void RdmaCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  endpoint_manager_->SendActorMsg(dst_machine_id, msg);
}

int8_t RdmaCommNet::IncreaseDoneCnt(ReadContext* read_ctx) {
  std::unique_lock<std::mutex> lck(read_ctx->done_cnt_mtx);
  read_ctx->done_cnt += 1;
  return read_ctx->done_cnt;
}

void RdmaCommNet::FinishOneReadContext(ActorReadContext* actor_read_ctx,
                                       ReadContext* read_ctx) {
  CHECK_EQ(actor_read_ctx->read_ctx_list.front(), read_ctx);
  actor_read_ctx->read_ctx_list.pop_front();
  for (std::function<void()>& callback : read_ctx->cbl) { callback(); }
}

}  // namespace oneflow
