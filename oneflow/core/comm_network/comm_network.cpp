#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

CommNet::~CommNet() {
  for (auto peer_mchn_id : peer_machine_id()) { CHECK(peer_mchn_id2wq_.at(peer_mchn_id).empty()); }
  ready_cbs_.Close();
  ready_cb_poller_.join();
}

void CommNet::Read(int64_t src_machine_id, void* src_token, void* dst_token) {
  ReadContext* read_ctx = new ReadContext(src_machine_id);
  auto do_read = [this, read_ctx, src_machine_id, src_token, dst_token]() {
    DoRead(read_ctx, src_machine_id, src_token, dst_token);
  };
  AddWorkToStream(read_ctx, do_read, true);
  last_read_ctx_ = read_ctx;
}

void CommNet::AddReadCallBack(std::function<void()> callback) {
  AddWorkToStream(last_read_ctx_, callback, false);
}

void CommNet::ReadDone(void* read_id) {
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  read_ctx->read_done = true;
  {
    auto& waiting_queue = peer_mchn_id2wq_.at(read_ctx->peer_mchn_id);
    std::unique_lock<std::mutex> lck(peer_mchn_id2wq_mtx_.at(read_ctx->peer_mchn_id));
    CHECK(!waiting_queue.empty() && waiting_queue.front().read_ctx->read_done);
    waiting_queue.pop();
    while (!waiting_queue.empty() && waiting_queue.front().read_ctx->read_done) {
      DoCallBack(waiting_queue.front().callback);
      waiting_queue.pop();
    }
    if (!waiting_queue.empty()) {
      auto waiting_item = waiting_queue.front();
      CHECK(waiting_item.is_read);
      CHECK_EQ(waiting_item.read_ctx->read_done, false);
      DoCallBack(waiting_item.callback);
      delete read_ctx;
    }
  }
}

void CommNet::DoCallBack(const std::function<void()>& cb) {
  CHECK(cb);
  ready_cbs_.Send(cb);
}

void CommNet::AddWorkToStream(ReadContext* read_ctx, const std::function<void()>& cb,
                              bool is_read) {
  std::unique_lock<std::mutex> lck(peer_mchn_id2wq_mtx_.at(read_ctx->peer_mchn_id));
  bool is_wq_empty = peer_mchn_id2wq_.at(read_ctx->peer_mchn_id).empty();
  if (is_wq_empty) { DoCallBack(cb); }
  if (!is_wq_empty || is_read) {
    peer_mchn_id2wq_.at(read_ctx->peer_mchn_id).push(CommNetItem(read_ctx, cb, is_read));
  }
}

CommNet::CommNet(const Plan& plan) : last_read_ctx_(nullptr) {
  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  HashMap<int64_t, MachineIds> net_topo;
  net_topo = PbMap2HashMap(plan.net_topo().peer_machine_ids());
  auto machine_ids_it = net_topo.find(this_machine_id);
  CHECK(machine_ids_it != net_topo.end());
  std::vector<int64_t> peer_machine_ids = PbRf2StdVec(machine_ids_it->second.machine_id());
  peer_machine_id_.insert(peer_machine_ids.begin(), peer_machine_ids.end());
  for (auto peer_mchn_id : peer_machine_id_) {
    peer_mchn_id2wq_mtx_[peer_mchn_id];
    CHECK(peer_mchn_id2wq_[peer_mchn_id].empty());
  }

  ready_cb_poller_ = std::thread([this]() {
    std::function<void()> cb;
    while (ready_cbs_.Receive(&cb) == kChannelStatusSuccess) { cb(); }
  });
}

}  // namespace oneflow
