#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

CommNet::~CommNet() {
  ready_cbs_.Close();
  ready_cb_poller_.join();
}

void* CommNet::NewActorReadId() const { return new ActorReadContext; }

void CommNet::DeleteActorReadId(void* actor_read_id) const {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->waiting_list.empty());
  delete actor_read_ctx;
}

void CommNet::Read(void* actor_read_id, int64_t src_machine_id, void* src_token, void* dst_token) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = new ReadContext;
  read_ctx->actor_read_ctx = actor_read_ctx;
  read_ctx->peer_mchn_id = src_machine_id;
  read_ctx->read_done = false;
  auto do_read = [this, read_ctx, src_machine_id, src_token, dst_token]() {
    DoRead(read_ctx, src_machine_id, src_token, dst_token);
  };
  {
    std::unique_lock<std::mutex> lck(peer_mchn_id2cq_mtx_.at(src_machine_id));
    peer_mchn_id2cq_.at(src_machine_id).push(read_ctx);
    AddWorkToStream(actor_read_id, do_read, true);
  }
}

void CommNet::AddReadCallBack(void* actor_read_id, std::function<void()> callback) {
  AddWorkToStream(actor_read_id, callback, false);
}

void CommNet::ReadDone(void* read_id) {
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  read_ctx->read_done = true;
  {
    std::unique_lock<std::mutex> lck(peer_mchn_id2cq_mtx_.at(read_ctx->peer_mchn_id));
    auto& completion_queue = peer_mchn_id2cq_.at(read_ctx->peer_mchn_id);
    while (!completion_queue.empty() && completion_queue.front()->read_done == true) {
      DoCallBack(completion_queue.front());
      completion_queue.pop();
    }
  }
}

void CommNet::DoCallBack(ReadContext* read_ctx) {
  ActorReadContext* actor_read_ctx = read_ctx->actor_read_ctx;
  CommNetItem item;
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->waiting_list_mtx);
    CHECK(!actor_read_ctx->waiting_list.empty());
    CHECK(actor_read_ctx->waiting_list.front().callback == nullptr);
    actor_read_ctx->waiting_list.pop_front();
  }
  while (true) {
    {
      std::unique_lock<std::mutex> lck(actor_read_ctx->waiting_list_mtx);
      if (actor_read_ctx->waiting_list.empty()
          || (actor_read_ctx->waiting_list.front().callback == nullptr)) {
        break;
      }
      item = actor_read_ctx->waiting_list.front();
      actor_read_ctx->waiting_list.pop_front();
    }
    CHECK(item.callback);
    ready_cbs_.Send(item.callback);
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
  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  HashMap<int64_t, MachineIds> net_topo;
  net_topo = PbMap2HashMap(plan.net_topo().peer_machine_ids());
  auto machine_ids_it = net_topo.find(this_machine_id);
  CHECK(machine_ids_it != net_topo.end());
  std::vector<int64_t> peer_machine_ids = PbRf2StdVec(machine_ids_it->second.machine_id());
  peer_machine_id_.insert(peer_machine_ids.begin(), peer_machine_ids.end());
  for (auto peer_mchn_id : peer_machine_id_) {
    peer_mchn_id2cq_mtx_[peer_mchn_id];
    CHECK(peer_mchn_id2cq_[peer_mchn_id].empty());
  }

  ready_cb_poller_ = std::thread([this]() {
    std::function<void()> cb;
    while (ready_cbs_.Receive(&cb) == kChannelStatusSuccess) { cb(); }
  });
}

}  // namespace oneflow
