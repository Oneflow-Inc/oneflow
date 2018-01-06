#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

CommNet* CommNet::comm_network_ptr_;

void* CommNet::NewActorReadId() { return new ActorReadContext; }

void CommNet::DeleteActorReadId(void* actor_read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->read_ctx_list.empty());
  delete actor_read_ctx;
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

void CommNet::AddReadCallBackDone(void* actor_read_id, void* read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if (IncreaseDoneCnt(read_ctx) == 2) {
    FinishOneReadContext(actor_read_ctx, read_ctx);
    delete read_ctx;
  }
}

void CommNet::ReadDone(void* read_done_id) {
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

int8_t CommNet::IncreaseDoneCnt(ReadContext* read_ctx) {
  std::unique_lock<std::mutex> lck(read_ctx->done_cnt_mtx);
  read_ctx->done_cnt += 1;
  return read_ctx->done_cnt;
}

void CommNet::FinishOneReadContext(ActorReadContext* actor_read_ctx,
                                   ReadContext* read_ctx) {
  CHECK_EQ(actor_read_ctx->read_ctx_list.front(), read_ctx);
  actor_read_ctx->read_ctx_list.pop_front();
  for (std::function<void()>& callback : read_ctx->cbl) { callback(); }
}

}  // namespace oneflow
