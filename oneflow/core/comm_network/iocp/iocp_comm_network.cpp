#include "oneflow/core/comm_network/iocp/iocp_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/runtime_context.h"

#ifdef PLATFORM_WINDOWS

namespace oneflow {

IOCPCommNet::~IOCPCommNet() {
  io_worker_ptr_->Stop();
  OF_BARRIER();
  delete io_worker_ptr_;
}

void IOCPCommNet::Init() {
  CommNet::Singleton()->set_comm_network_ptr(new IOCPCommNet());
}

const void* IOCPCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  auto mem_desc = new SocketMemDesc;
  mem_desc->mem_ptr = mem_ptr;
  mem_desc->byte_size = byte_size;
  {
    std::unique_lock<std::mutex> lck(mem_desc_mtx_);
    mem_descs_.push_back(mem_desc);
  }
  return mem_desc;
}

void IOCPCommNet::UnRegisterMemory(const void* token) {
  std::unique_lock<std::mutex> lck(mem_desc_mtx_);
  CHECK(!mem_descs_.empty());
  unregister_mem_descs_cnt_ += 1;
  if(unregister_mem_descs_cnt_ == mem_descs_.size()) {
    for(SocketMemDesc* mem_desc : mem_descs_) { delete mem_desc; }
    mem_descs_.clear();
    unregister_mem_descs_cnt_ = 0;
  }
}

void IOCPCommNet::RegisterMemoryDone() {
  // do nothing
}

void* IOCPCommNet::NewActorReadId() { return new ActorReadContext; }

void IOCPCommNet::DeleteActorReadId(void* actor_read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->read_ctx_list.empty());
  delete actor_read_ctx;
}

void* IOCPCommNet::Read(void* actor_read_id, int64_t write_machine_id,
                         const void* write_token, const void* read_token) {
  // ReadContext
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = new ReadContext;
  read_ctx->done_cnt = 0;
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    actor_read_ctx->read_ctx_list.push_back(read_ctx);
  }
  // request write msg
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kRequestWrite;
  msg.socket_token.write_machine_mem_desc_ = write_token;
  msg.socket_token.read_machine_mem_desc_ = read_token;
  msg.socket_token.read_done_id = new ReadDoneContext(actor_read_ctx, read_ctx);
  io_worker_ptr_->PostSendMsgRequest(write_machine_id, msg);
  return read_ctx;
}

void IOCPCommNet::AddReadCallBack(void* actor_read_id, void* read_id,
                                   std::function<void()> callback) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if(read_ctx) {
    read_ctx->cbl.push_back(callback);
    return;
  }
  do {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    if(actor_read_ctx->read_ctx_list.empty()) {
      break;
    } else {
      actor_read_ctx->read_ctx_list.back()->cbl.push_back(callback);
      return;
    }
  } while(0);
  callback();
}

void IOCPCommNet::AddReadCallBackDone(void* actor_read_id, void* read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if(IncreaseDoneCnt(read_ctx) == 2) {
    FinishOneReadContext(actor_read_ctx, read_ctx);
    delete read_ctx;
  }
}

void IOCPCommNet::ReadDone(void* read_done_id) {
  auto parsed_read_done_id = static_cast<ReadDoneContext*>(read_done_id);
  auto actor_read_ctx = std::get<0>(*parsed_read_done_id);
  auto read_ctx = std::get<1>(*parsed_read_done_id);
  delete parsed_read_done_id;
  if(IncreaseDoneCnt(read_ctx) == 2) {
    {
      std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
      FinishOneReadContext(actor_read_ctx, read_ctx);
    }
    delete read_ctx;
  }
}

int8_t IOCPCommNet::IncreaseDoneCnt(ReadContext* read_ctx) {
  std::unique_lock<std::mutex> lck(read_ctx->done_cnt_mtx);
  read_ctx->done_cnt += 1;
  return read_ctx->done_cnt;
}

void IOCPCommNet::FinishOneReadContext(ActorReadContext* actor_read_ctx,
                                        ReadContext* read_ctx) {
  CHECK_EQ(actor_read_ctx->read_ctx_list.front(), read_ctx);
  actor_read_ctx->read_ctx_list.pop_front();
  for(std::function<void()>& callback : read_ctx->cbl) { callback(); }
}

void IOCPCommNet::SendActorMsg(int64_t dst_machine_id,
                                const ActorMsg& actor_msg) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kActor;
  msg.actor_msg = actor_msg;
  io_worker_ptr_->PostSendMsgRequest(dst_machine_id, msg);
}

IOCPCommNet::IOCPCommNet() {
  mem_descs_.clear();
  unregister_mem_descs_cnt_ = 0;
  io_worker_ptr_ = new IOWorker();
}

}  // namespace oneflow

#endif // PLATFORM_WINDOWS
