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
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/env_desc.h"

namespace oneflow {

namespace {

const int32_t max_retry_num = 60;
const int64_t sleep_seconds = 10;

#define GRPC_CHECK(x) CHECK_EQ(x.error_code(), grpc::StatusCode::OK)

template<CtrlMethod ctrl_method>
class ClientCall final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClientCall);
  ClientCall() = default;
  ~ClientCall() = default;

  CtrlRequest<ctrl_method>* mut_request() { return &request_; }
  const CtrlResponse<ctrl_method>& response() const { return response_; }
  void operator()(CtrlService::Stub* stub) {
    grpc::ClientContext client_ctx;
    GRPC_CHECK(stub->CallMethod<ctrl_method>(&client_ctx, request_, &response_));
  }

 private:
  CtrlRequest<ctrl_method> request_;
  CtrlResponse<ctrl_method> response_;
};

}  // namespace

CtrlClient::~CtrlClient() {
  {
    std::unique_lock<std::mutex> lck(need_heartbeat_thread_stop_mtx_);
    need_heartbeat_thread_stop_ = true;
  }
  heartbeat_thread_.join();
}

void CtrlClient::Barrier(const std::string& barrier_name) {
  Barrier(barrier_name, Global<EnvDesc>::Get()->TotalMachineNum());
}

void CtrlClient::Barrier(const std::string& barrier_name, int32_t barrier_num) {
  ClientCall<CtrlMethod::kBarrier> call;
  call.mut_request()->set_name(barrier_name);
  call.mut_request()->set_num(barrier_num);
  call(GetMasterStub());
}

TryLockResult CtrlClient::TryLock(const std::string& name) {
  {
    std::unique_lock<std::mutex> lck(done_names_mtx_);
    if (done_names_.find(name) != done_names_.end()) { return TryLockResult::kDone; }
  }
  ClientCall<CtrlMethod::kTryLock> call;
  call.mut_request()->set_name(name);
  call(GetResponsibleStub(name));
  if (call.response().result() == TryLockResult::kDone) {
    std::unique_lock<std::mutex> lck(done_names_mtx_);
    done_names_.insert(name);
  }
  return call.response().result();
}

void CtrlClient::NotifyDone(const std::string& name) {
  ClientCall<CtrlMethod::kNotifyDone> call;
  call.mut_request()->set_name(name);
  call(GetResponsibleStub(name));
}

void CtrlClient::WaitUntilDone(const std::string& name) {
  ClientCall<CtrlMethod::kWaitUntilDone> call;
  call.mut_request()->set_name(name);
  call(GetResponsibleStub(name));
}

void CtrlClient::PushKV(const std::string& k, std::function<void(std::string*)> VSetter) {
  ClientCall<CtrlMethod::kPushKV> call;
  call.mut_request()->set_key(k);
  VSetter(call.mut_request()->mutable_val());
  call(GetResponsibleStub(k));
}

void CtrlClient::PushMasterKV(const std::string& k, std::function<void(std::string*)> VSetter) {
  ClientCall<CtrlMethod::kPushKV> call;
  call.mut_request()->set_key(k);
  VSetter(call.mut_request()->mutable_val());
  call(GetMasterStub());
}

void CtrlClient::PushKV(const std::string& k, const std::string& v) {
  PushKV(k, [&](std::string* o) { *o = v; });
}

void CtrlClient::PushKV(const std::string& k, const PbMessage& msg) {
  PushKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void CtrlClient::PushMasterKV(const std::string& k, const PbMessage& msg) {
  PushMasterKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void CtrlClient::ClearKV(const std::string& k) {
  ClientCall<CtrlMethod::kClearKV> call;
  call.mut_request()->set_key(k);
  call(GetResponsibleStub(k));
}

void CtrlClient::ClearMasterKV(const std::string& k) {
  ClientCall<CtrlMethod::kClearKV> call;
  call.mut_request()->set_key(k);
  call(GetMasterStub());
}

void CtrlClient::PullKV(const std::string& k, std::function<void(const std::string&)> VGetter) {
  ClientCall<CtrlMethod::kPullKV> call;
  call.mut_request()->set_key(k);
  call(GetResponsibleStub(k));
  VGetter(call.response().val());
}

void CtrlClient::PullMasterKV(const std::string& k,
                              std::function<void(const std::string&)> VGetter) {
  ClientCall<CtrlMethod::kPullKV> call;
  call.mut_request()->set_key(k);
  call(GetMasterStub());
  VGetter(call.response().val());
}

void CtrlClient::PullKV(const std::string& k, std::string* v) {
  PullKV(k, [&](const std::string& i) { *v = i; });
}

void CtrlClient::PullKV(const std::string& k, PbMessage* msg) {
  PullKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void CtrlClient::PullMasterKV(const std::string& k, PbMessage* msg) {
  PullMasterKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void CtrlClient::PushActEvent(const ActEvent& act_event) {
  ClientCall<CtrlMethod::kPushActEvent> call;
  *(call.mut_request()->mutable_act_event()) = act_event;
  call(GetMasterStub());
}

void CtrlClient::Clear() {
  ClientCall<CtrlMethod::kClear> call;
  call(GetThisStub());
  std::unique_lock<std::mutex> lck(done_names_mtx_);
  done_names_.clear();
}

int32_t CtrlClient::IncreaseCount(const std::string& k, int32_t v) {
  ClientCall<CtrlMethod::kIncreaseCount> call;
  call.mut_request()->set_key(k);
  call.mut_request()->set_val(v);
  call(GetResponsibleStub(k));
  return call.response().val();
}

void CtrlClient::EraseCount(const std::string& k) {
  ClientCall<CtrlMethod::kEraseCount> call;
  call.mut_request()->set_key(k);
  call(GetResponsibleStub(k));
}

CtrlClient::CtrlClient() {
  stubs_.reserve(Global<EnvDesc>::Get()->TotalMachineNum());
  int32_t port = -1;
  std::string addr = "";
  for (int64_t i = 0; i < Global<EnvDesc>::Get()->TotalMachineNum(); ++i) {
    const Machine& mchn = Global<EnvDesc>::Get()->machine(i);
    port = (mchn.ctrl_port_agent() != -1) ? (mchn.ctrl_port_agent())
                                          : Global<EnvDesc>::Get()->ctrl_port();
    addr = mchn.addr() + ":" + std::to_string(port);
    stubs_.push_back(CtrlService::NewStub(addr));
    LoadServer(mchn.addr(), stubs_[i].get());
  }
  need_heartbeat_thread_stop_ = false;
  heartbeat_thread_ = std::thread([this]() {
    std::mt19937 gen(NewRandomSeed());
    std::uniform_int_distribution<int32_t> sleep_second_dis(7, 13);
    LoadServerRequest request;
    LoadServerResponse response;
    while (true) {
      {
        std::unique_lock<std::mutex> lck(need_heartbeat_thread_stop_mtx_);
        if (need_heartbeat_thread_stop_) { break; }
      }
      for (size_t i = 0; i < stubs_.size(); ++i) {
        grpc::ClientContext client_ctx;
        request.set_addr(Global<EnvDesc>::Get()->machine(i).addr());
        GRPC_CHECK(stubs_[i]->CallMethod<CtrlMethod::kLoadServer>(&client_ctx, request, &response))
            << "Machine " << i << " lost";
      }
      std::this_thread::sleep_for(std::chrono::seconds(sleep_second_dis(gen)));
    }
  });
}

void CtrlClient::LoadServer(const std::string& server_addr, CtrlService::Stub* stub) {
  int32_t retry_idx = 0;
  for (; retry_idx < max_retry_num; ++retry_idx) {
    grpc::ClientContext client_ctx;
    LoadServerRequest request;
    request.set_addr(server_addr);
    LoadServerResponse response;
    grpc::Status st = stub->CallMethod<CtrlMethod::kLoadServer>(&client_ctx, request, &response);
    if (st.error_code() == grpc::StatusCode::OK) {
      LOG(INFO) << "LoadServer " << server_addr << " Successful at " << retry_idx << " times";
      break;
    } else if (st.error_code() == grpc::StatusCode::UNAVAILABLE) {
      LOG(INFO) << "LoadServer " << server_addr << " Failed at " << retry_idx << " times";
      std::this_thread::sleep_for(std::chrono::seconds(sleep_seconds));
      continue;
    } else {
      LOG(FATAL) << st.error_message();
    }
  }
  CHECK_LT(retry_idx, max_retry_num);
}

CtrlService::Stub* CtrlClient::GetThisStub() {
  return stubs_[Global<MachineCtx>::Get()->this_machine_id()].get();
}

CtrlService::Stub* CtrlClient::GetResponsibleStub(const std::string& key) {
  int64_t machine_id = (std::hash<std::string>{}(key)) % Global<EnvDesc>::Get()->TotalMachineNum();
  return stubs_[machine_id].get();
}

}  // namespace oneflow
