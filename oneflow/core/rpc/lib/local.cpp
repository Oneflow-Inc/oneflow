/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifdef RPC_BACKEND_LOCAL

#include "glog/logging.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/rpc/include/local.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

LocalCtrlClient::LocalCtrlClient(const ProcessCtx& process_ctx) {
  CHECK(process_ctx.ctrl_addr_size() == 1);
  CHECK(process_ctx.node_size() == 1);
}

void LocalCtrlClient::Barrier(const std::string& barrier_name) {
  Barrier(barrier_name, Global<EnvDesc>::Get()->TotalMachineNum());
}

void LocalCtrlClient::Barrier(const std::string& barrier_name, int32_t barrier_num) {
  CHECK(barrier_num == 1);
}

TryLockResult LocalCtrlClient::TryLock(const std::string& name) {
  std::unique_lock<std::mutex> lck(done_names_mtx_);
  if (done_names_.find(name) != done_names_.end()) {
    return TryLockResult::kDone;
  } else if (doing_names_.find(name) != doing_names_.end()) {
    return TryLockResult::kDoing;
  } else {
    doing_names_.insert(name);
    return TryLockResult::kLocked;
  }
}

void LocalCtrlClient::NotifyDone(const std::string& name) {
  std::unique_lock<std::mutex> lck(done_names_mtx_);
  done_names_.insert(name);
  CHECK_EQ(doing_names_.erase(name), 1);
  done_names_cv_.notify_all();
}

void LocalCtrlClient::WaitUntilDone(const std::string& name) {
  std::unique_lock<std::mutex> lck(done_names_mtx_);
  LOG(INFO) << "waiting for name: " << name;
  done_names_cv_.wait(lck);
  CHECK(done_names_.find(name) != done_names_.end());
}

void LocalCtrlClient::PushKV(const std::string& k, std::function<void(std::string*)> VSetter) {
  std::unique_lock<std::mutex> lck(kv_mtx_);
  VSetter(&kv_[k]);
  kv_cv_.notify_all();
}

void LocalCtrlClient::PushKV(const std::string& k, const std::string& v) {
  PushKV(k, [&](std::string* o) { *o = v; });
}

void LocalCtrlClient::PushKV(const std::string& k, const PbMessage& msg) {
  PushKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void LocalCtrlClient::PushMasterKV(const std::string& k, const PbMessage& msg) {
  PushKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void LocalCtrlClient::ClearKV(const std::string& k) {
  std::unique_lock<std::mutex> lck(kv_mtx_);
  kv_.erase(k);
}

void LocalCtrlClient::ClearMasterKV(const std::string& k) { ClearKV(k); }

void LocalCtrlClient::PullKV(const std::string& k,
                             std::function<void(const std::string&)> VGetter) {
  std::unique_lock<std::mutex> lck(kv_mtx_);
  while (true) {
    auto it = kv_.find(k);
    if (it == kv_.end()) {
      LOG(INFO) << "waiting for key: " << k;
      kv_cv_.wait(lck);
    } else {
      VGetter(it->second);
      break;
    }
  }
}

void LocalCtrlClient::PullKV(const std::string& k, std::string* v) {
  PullKV(k, [&](const std::string& i) { *v = i; });
}

void LocalCtrlClient::PullKV(const std::string& k, PbMessage* msg) {
  PullKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void LocalCtrlClient::PullMasterKV(const std::string& k, PbMessage* msg) {
  PullKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void LocalCtrlClient::Clear() {
  {
    std::unique_lock<std::mutex> lck(done_names_mtx_);
    done_names_.clear();
    done_names_cv_.notify_all();
  }
  {
    std::unique_lock<std::mutex> lck(kv_mtx_);
    kv_.clear();
    kv_cv_.notify_all();
  }
}

int32_t LocalCtrlClient::IncreaseCount(const std::string& k, int32_t v) {
  std::unique_lock<std::mutex> lck(counter_mtx_);
  auto it = counter_.find(k);
  if (it == counter_.end()) {
    counter_[k] = 1;
    return 1;
  } else {
    const int32_t new_val = it->second + 1;
    counter_[k] = new_val;
    return new_val;
  }
}

void LocalCtrlClient::EraseCount(const std::string& k) {
  std::unique_lock<std::mutex> lck(counter_mtx_);
  counter_.erase(k);
}

class DryRunCtrlClient : public CtrlClient {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DryRunCtrlClient);
  explicit DryRunCtrlClient(const ProcessCtx& process_ctx)
      : local_ctrl_client_{std::unique_ptr<LocalCtrlClient>(new LocalCtrlClient(process_ctx))} {
    CHECK(process_ctx.ctrl_addr_size() == 1);
    CHECK(process_ctx.node_size() == 1);
  }
  ~DryRunCtrlClient() override = default;

  void Barrier(const std::string& barrier_name) override {
    Barrier(barrier_name, Global<EnvDesc>::Get()->TotalMachineNum());
  }
  void Barrier(const std::string& barrier_name, int32_t barrier_num) override {
    LOG(INFO) << "skipping barrier in dry run, barrier name: " << barrier_name
              << ", barrier num: " << barrier_num;
  }

  TryLockResult TryLock(const std::string& name) override {
    return local_ctrl_client_->TryLock(name);
  }
  void NotifyDone(const std::string& name) override { local_ctrl_client_->NotifyDone(name); }
  void WaitUntilDone(const std::string& name) override { local_ctrl_client_->WaitUntilDone(name); }

  void PushKV(const std::string& k, std::function<void(std::string*)> VSetter) override {
    local_ctrl_client_->PushKV(k, VSetter);
  }
  void PushKV(const std::string& k, const std::string& v) override {
    local_ctrl_client_->PushKV(k, v);
  }
  void PushKV(const std::string& k, const PbMessage& msg) override {
    local_ctrl_client_->PushKV(k, msg);
  }
  void PushMasterKV(const std::string& k, const PbMessage& msg) override {
    local_ctrl_client_->PushMasterKV(k, msg);
  }

  void ClearKV(const std::string& k) override { local_ctrl_client_->ClearKV(k); }
  void ClearMasterKV(const std::string& k) override { local_ctrl_client_->ClearMasterKV(k); }

  void PullKV(const std::string& k, std::function<void(const std::string&)> VGetter) override {
    local_ctrl_client_->PullKV(k, VGetter);
  }
  void PullKV(const std::string& k, std::string* v) override { local_ctrl_client_->PullKV(k, v); }
  void PullKV(const std::string& k, PbMessage* msg) override { local_ctrl_client_->PullKV(k, msg); }
  void PullMasterKV(const std::string& k, PbMessage* msg) override {
    local_ctrl_client_->PullMasterKV(k, msg);
  }
  void PushActEvent(const ActEvent& ev) override { local_ctrl_client_->PushActEvent(ev); }
  void Clear() override { local_ctrl_client_->Clear(); }
  int32_t IncreaseCount(const std::string& k, int32_t v) override {
    return local_ctrl_client_->IncreaseCount(k, v);
  }
  void EraseCount(const std::string& k) override { local_ctrl_client_->EraseCount(k); }

 private:
  std::unique_ptr<LocalCtrlClient> local_ctrl_client_;
};

void SetLocalProcessCtx(oneflow::ProcessCtx* ctx) {
  Address* addr = ctx->add_ctrl_addr();
  addr->set_host("localhost");
  ctx->set_rank(0);
  ctx->set_node_size(1);
}

Maybe<void> LocalRpcManager::Bootstrap() {
  SetLocalProcessCtx(Global<ProcessCtx>::Get());
  return Maybe<void>::Ok();
}

Maybe<void> LocalRpcManager::CreateClient() {
  auto* client = new LocalCtrlClient(*Global<ProcessCtx>::Get());
  Global<CtrlClient>::SetAllocated(client);
  return Maybe<void>::Ok();
}

LocalRpcManager::~LocalRpcManager() { Global<CtrlClient>::Delete(); }

Maybe<void> DryRunRpcManager::Bootstrap() {
  SetLocalProcessCtx(Global<ProcessCtx>::Get());
  return Maybe<void>::Ok();
}

Maybe<void> DryRunRpcManager::CreateClient() {
  auto* client = new DryRunCtrlClient(*Global<ProcessCtx>::Get());
  Global<CtrlClient>::SetAllocated(client);
  return Maybe<void>::Ok();
}

DryRunRpcManager::~DryRunRpcManager() { Global<CtrlClient>::Delete(); }

}  // namespace oneflow

#endif  // RPC_BACKEND_LOCAL
