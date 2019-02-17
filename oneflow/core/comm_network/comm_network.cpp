#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

CommNet::~CommNet() {
  ready_cbs_.Close();
  ready_cb_poller_.join();
}

void* CommNet::NewActorReadId() const { return new ActorReadContext; }

void CommNet::DeleteActorReadId(void* actor_read_id) const {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->read_ctx_list.empty());
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
  read_ctx->do_read = do_read;
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    actor_read_ctx->read_ctx_list.push_back(read_ctx);
  }
  {
    std::unique_lock<std::mutex> lck(peer_mchn_id2cq_mtx_.at(src_machine_id));
    if (peer_mchn_id2cq_.at(src_machine_id).empty()) { ready_cbs_.Send(do_read); }
    peer_mchn_id2cq_.at(src_machine_id).push(read_ctx);
  }
}

void CommNet::AddReadCallBack(void* actor_read_id, std::function<void()> callback) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
  if (actor_read_ctx->read_ctx_list.empty()) {
    ready_cbs_.Send(callback);
  } else {
    std::unique_lock<std::mutex> lck(actor_read_ctx->waiting_list_mtx);
    auto read_ctx = actor_read_ctx->read_ctx_list.back();
    actor_read_ctx->waiting_list.push_back(std::make_pair(read_ctx, callback));
  }
}

void CommNet::ReadDone(void* read_id) {
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  read_ctx->read_done = true;
  std::unique_lock<std::mutex> lck(peer_mchn_id2cq_mtx_.at(read_ctx->peer_mchn_id));
  auto& completion_queue = peer_mchn_id2cq_.at(read_ctx->peer_mchn_id);
  while (!completion_queue.empty() && completion_queue.front()->read_done == true) {
    DoCallBack(completion_queue.front());
    completion_queue.pop();
  }
  if (!completion_queue.empty()) {
    CHECK_EQ(completion_queue.front()->read_done, false);
    ready_cbs_.Send(completion_queue.front()->do_read);
  }
}

void CommNet::DoCallBack(ReadContext* read_ctx) {
  ActorReadContext* actor_read_ctx = read_ctx->actor_read_ctx;
  std::unique_lock<std::mutex> lck_read_ctx_list(actor_read_ctx->read_ctx_list_mtx);
  CHECK_EQ(read_ctx, actor_read_ctx->read_ctx_list.front());
  std::unique_lock<std::mutex> lck_waiting_list(actor_read_ctx->waiting_list_mtx);
  auto& waiting_list = actor_read_ctx->waiting_list;
  while (!waiting_list.empty() && waiting_list.front().first == read_ctx) {
    ready_cbs_.Send(waiting_list.front().second);
    waiting_list.pop_front();
  }
  actor_read_ctx->read_ctx_list.pop_front();
  delete read_ctx;
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
