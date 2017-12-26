#include "oneflow/core/comm_network/rdma/rdma_comm_network.h"
#include "oneflow/core/job/machine_context.h"
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

void RdmaCommNet::EstablishNetwork() {
  endpoint_manager_->InitRdma();
  endpoint_manager_->Start();
}

RdmaCommNet::RdmaCommNet() {
  mems_.clear();
  unregister_mems_cnt_ = 0;
  endpoint_manager_ = new EndpointManager();
}

RdmaCommNet::~RdmaCommNet() {
  endpoint_manager_->Stop();
  delete endpoint_manager_;
  CHECK(mems_.empty());
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
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
  int64_t this_machine_id = MachineCtx::Singleton()->this_machine_id();
  HashMap<uint64_t, RdmaMemDesc> this_machine_tokens_msg;
  for (RdmaMem* mem_ptr : mems_) {
    this_machine_tokens_msg.insert(
        {reinterpret_cast<uint64_t>(mem_ptr), mem_ptr->GetRdmaMemDesc()});
  }
  RdmaTokensMsg tokens_msg;
  *(tokens_msg.mutable_token2mem_desc()) =
      HashMap2PbMap<uint64_t, RdmaMemDesc>(this_machine_tokens_msg);
  CtrlClient::Singleton()->PushKV(GenTokensMsgKey(this_machine_id), tokens_msg);
  OF_BARRIER();
  FOR_RANGE(int64_t, peer_machine_id, 0, total_machine_num) {
    if (peer_machine_id == MachineCtx::Singleton()->this_machine_id()) {
      continue;
    }
    RdmaTokensMsg peer_tokens_msg;
    CtrlClient::Singleton()->PullKV(GenTokensMsgKey(peer_machine_id),
                                    &peer_tokens_msg);
    HashMap<uint64_t, RdmaMemDesc> peer_token2mem_desc =
        PbMap2HashMap(peer_tokens_msg.token2mem_desc());
    for (auto pair : peer_token2mem_desc) {
      token2mem_desc_.insert({pair.first, pair.second});
    }
  }
  OF_BARRIER();
  LOG(INFO) << "Finish RegisterMemoryDone";
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
  // LOG(INFO) << "SendActorMsg start, MsgTye: " << msg.msg_type()
  //           << ", Msg.dst_actor_id:" << msg.dst_actor_id();
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
