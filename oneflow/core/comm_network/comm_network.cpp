#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

CommNet* CommNet::comm_network_ptr_;

void* CommNet::NewActorReadId() { return new ActorReadContext; }

void CommNet::DeleteActorReadId(void* actor_read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->read_ctx_list.empty());
  delete actor_read_ctx;
}

void* CommNet::Read(void* actor_read_id, int64_t src_machine_id,
                    const void* src_token, const void* dst_token) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = new ReadContext;
  read_ctx->actor_read_ctx = actor_read_ctx;
  read_ctx->done_cnt = 0;
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    actor_read_ctx->read_ctx_list.push_back(read_ctx);
  }
  DoRead(read_ctx, src_machine_id, src_token, dst_token);
  return read_ctx;
}

void CommNet::AddReadCallBack(void* actor_read_id, void* read_id,
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

void CommNet::AddReadCallBackDone(void* read_id) {
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if (IncreaseDoneCnt(read_ctx) == 2) { FinishOneRead(read_ctx); }
}

void CommNet::ReadDone(void* read_id) {
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if (IncreaseDoneCnt(read_ctx) == 2) {
    std::unique_lock<std::mutex> lck(
        read_ctx->actor_read_ctx->read_ctx_list_mtx);
    FinishOneRead(read_ctx);
  }
}

int8_t CommNet::IncreaseDoneCnt(ReadContext* read_ctx) {
  std::unique_lock<std::mutex> lck(read_ctx->done_cnt_mtx);
  read_ctx->done_cnt += 1;
  return read_ctx->done_cnt;
}

void CommNet::FinishOneRead(ReadContext* read_ctx) {
  ActorReadContext* actor_read_ctx = read_ctx->actor_read_ctx;
  CHECK_EQ(actor_read_ctx->read_ctx_list.front(), read_ctx);
  actor_read_ctx->read_ctx_list.pop_front();
  for (std::function<void()>& callback : read_ctx->cbl) { callback(); }
  delete read_ctx;
}

}  // namespace oneflow
