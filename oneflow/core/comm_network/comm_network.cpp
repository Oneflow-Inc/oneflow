#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

CommNet::~CommNet() {
  for (int64_t stream_id = 0; stream_id < peer_machine_id_.size(); ++stream_id) {
    CHECK(stream_id2stream_.at(stream_id).empty());
  }
  ready_cbs_.Close();
  ready_cb_poller_.join();
}

void CommNet::Read(int64_t stream_id, int64_t src_machine_id, void* src_token, void* dst_token) {
  ReadContext* read_ctx = new ReadContext(stream_id);
  auto do_read = [this, read_ctx, src_machine_id, src_token, dst_token]() {
    DoRead(read_ctx, src_machine_id, src_token, dst_token);
  };
  AddWorkToStream(stream_id, do_read, true);
}

void CommNet::AddReadCallBack(int64_t stream_id, std::function<void()> callback) {
  AddWorkToStream(stream_id, callback, false);
}

/*
void CommNet::ReadDone(void* read_id) {
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  auto& local_stream = stream_id2stream_.at(read_ctx->stream_id);
  {
    std::unique_lock<std::mutex> lck(*stream_id2stream_mtx_ptr_.at(read_ctx->stream_id));
    CHECK(!local_stream.empty() && local_stream.front().is_read);
    local_stream.pop();
    while (!local_stream.empty() && !local_stream.front().is_read) {
      IssueCallBack(local_stream.front().callback);
      local_stream.pop();
    }
    if (!local_stream.empty()) {
      auto item = local_stream.front();
      CHECK(item.is_read);
      IssueCallBack(item.callback);
    }
  }
  delete read_ctx;
}
*/

void CommNet::ReadDone(void* read_id) {
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  int64_t stream_id = read_ctx->stream_id;
  auto& local_stream = stream_id2stream_.at(stream_id);
  {
    std::unique_lock<std::mutex> lck(*stream_id2stream_mtx_ptr_.at(stream_id));
    CHECK(!local_stream.empty() && local_stream.front().is_read);
    local_stream.pop();
  }
  while (true) {
    {
      std::unique_lock<std::mutex> lck(*stream_id2stream_mtx_ptr_.at(stream_id));
      if (local_stream.empty() || local_stream.front().is_read) { break; }
    }
    IssueCallBack(local_stream.front().callback);
    {
      std::unique_lock<std::mutex> lck(*stream_id2stream_mtx_ptr_.at(stream_id));
      local_stream.pop();
    }
  }
  delete read_ctx;
  {
    std::unique_lock<std::mutex> lck(*stream_id2stream_mtx_ptr_.at(stream_id));
    if (local_stream.empty()) { return; }
  }
  auto item = local_stream.front();
  CHECK(item.is_read);
  IssueCallBack(item.callback);
}

void CommNet::IssueCallBack(const std::function<void()>& cb) {
  CHECK(cb);
  ready_cbs_.Send(cb);
}

/*
void CommNet::AddWorkToStream(int64_t stream_id, const std::function<void()>& cb, bool is_read) {
  CHECK_LT(stream_id, peer_machine_id_.size());
  std::unique_lock<std::mutex> lck(*stream_id2stream_mtx_ptr_.at(stream_id));
  bool is_stream_empty = stream_id2stream_.at(stream_id).empty();
  if (is_stream_empty) { IssueCallBack(cb); }
  if (!is_stream_empty || is_read) {
    stream_id2stream_.at(stream_id).push(CommNetItem(cb, is_read));
  }
}
*/

void CommNet::AddWorkToStream(int64_t stream_id, const std::function<void()>& cb, bool is_read) {
  CHECK_LT(stream_id, peer_machine_id_.size());
  bool is_stream_empty;
  {
    std::unique_lock<std::mutex> lck(*stream_id2stream_mtx_ptr_.at(stream_id));
    is_stream_empty = stream_id2stream_.at(stream_id).empty();
  }
  if (is_stream_empty) { IssueCallBack(cb); }
  {
    std::unique_lock<std::mutex> lck(*stream_id2stream_mtx_ptr_.at(stream_id));
    if (!stream_id2stream_.at(stream_id).empty() || is_read) {
      stream_id2stream_.at(stream_id).push(CommNetItem(cb, is_read));
    }
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
  HashSet<int64_t> stream_ids;
  for (const auto& task : plan.task()) {
    if (task.machine_id() != this_machine_id) { continue; }
    if (task.task_type() != TaskType::kCopyCommNet) { continue; }
    stream_ids.emplace(Global<IDMgr>::Get()->LocalWorkStreamId4TaskId(task.task_id()));
  }
  for (int64_t stream_id : stream_ids) {
    CHECK(stream_id2stream_mtx_ptr_.emplace(stream_id, std::make_unique<std::mutex>()).second);
    CHECK(stream_id2stream_.emplace(stream_id, std::queue<CommNetItem>()).second);
    CHECK(stream_id2stream_.at(stream_id).empty());
  }

  ready_cb_poller_ = std::thread([this]() {
    std::function<void()> cb;
    while (ready_cbs_.Receive(&cb) == kChannelStatusSuccess) { cb(); }
  });
}

}  // namespace oneflow
