#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

CommNet::~CommNet() {
  ready_cbs_.CloseSendEnd();
  ready_cb_poller_.join();
}

void* CommNet::NewActorReadId() { return new ActorReadContext; }

void CommNet::DeleteActorReadId(void* actor_read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->waiting_list.empty());
  delete actor_read_ctx;
}

void CommNet::Read(void* actor_read_id, int64_t src_machine_id, void* src_token, void* dst_token) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = new ReadContext;
  read_ctx->actor_read_ctx = actor_read_ctx;
  auto do_read = [this, read_ctx, src_machine_id, src_token, dst_token]() {
    DoRead(read_ctx, src_machine_id, src_token, dst_token);
  };
  AddWorkToStream(actor_read_id, do_read, true);
}

void CommNet::AddReadCallBack(void* actor_read_id, std::function<void()> callback) {
  AddWorkToStream(actor_read_id, callback, false);
}

void CommNet::ReadDone(void* read_id) {
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  ActorReadContext* actor_read_ctx = read_ctx->actor_read_ctx;
  CommNetItem item;
  for (;;) {
    {
      std::unique_lock<std::mutex> lck(actor_read_ctx->waiting_list_mtx);
      if (actor_read_ctx->waiting_list.empty()) { break; }
      item = actor_read_ctx->waiting_list.front();
      actor_read_ctx->waiting_list.pop_front();
    }
    if (item.callback) { ready_cbs_.Send(item.callback); }
    if (item.is_read) { break; }
  }
  delete read_ctx;
}

void CommNet::AddWorkToStream(void* actor_read_id, const std::function<void()>& cb, bool is_read) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  std::unique_lock<std::mutex> lck(actor_read_ctx->waiting_list_mtx);
  if (actor_read_ctx->waiting_list.empty()) {
    ready_cbs_.Send(cb);
  } else {
    CommNetItem work_item(is_read, cb);
    actor_read_ctx->waiting_list.push_back(work_item);
  }
  if (is_read) {
    CommNetItem empty_cb;
    actor_read_ctx->waiting_list.push_back(empty_cb);
  }
}

CommNet::CommNet(const Plan& plan) {
  HashMap<int64_t, int64_t> rid2mid;
  HashMap<int64_t, int64_t> tid2mid;
  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();

  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& regst_desc_it : task_proto.produced_regst_desc()) {
      rid2mid.emplace(regst_desc_it.second.regst_desc_id(), task_proto.machine_id());
    }
    CHECK(tid2mid.emplace(task_proto.task_id(), task_proto.machine_id()).second);
  }
  for (const TaskProto& task_proto : plan.task()) {
    if (task_proto.machine_id() != this_machine_id) { continue; }
    for (const auto& regst_desc_set_it : task_proto.consumed_regst_desc_id()) {
      for (int64_t regst_desc_id : regst_desc_set_it.second.regst_desc_id()) {
        auto rid2mid_it = rid2mid.find(regst_desc_id);
        CHECK(rid2mid_it != rid2mid.end());
        peer_machine_id_.insert(rid2mid_it->second);
      }
    }
    for (const auto& regst_desc_it : task_proto.produced_regst_desc()) {
      for (int64_t consumer_task_id : regst_desc_it.second.consumer_task_id()) {
        auto tid2mid_it = tid2mid.find(consumer_task_id);
        CHECK(tid2mid_it != tid2mid.end());
        peer_machine_id_.insert(tid2mid_it->second);
      }
    }
  }
  peer_machine_id_.erase(this_machine_id);

  ready_cb_poller_ = std::thread([this]() {
    std::function<void()> cb;
    while (ready_cbs_.Receive(&cb) == 0) { cb(); }
  });
}

}  // namespace oneflow
