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

namespace oneflow {

namespace {

#define GRPC_CHECK(x) CHECK_EQ(x.error_code(), grpc::StatusCode::OK)

}  // namespace

GrpcCtrlClient::~GrpcCtrlClient() {
  {
    std::unique_lock<std::mutex> lck(need_heartbeat_thread_stop_mtx_);
    need_heartbeat_thread_stop_ = true;
  }
  heartbeat_thread_.join();
}

GrpcCtrlClient::GrpcCtrlClient(const ProcessCtx& process_ctx) : process_ctx_(process_ctx) {
  rpc_client_.ReserveStubsOfSize(process_ctx.ctrl_addr_size());
  for (int64_t i = 0; i < process_ctx.ctrl_addr_size(); ++i) {
    const Address& address = process_ctx.ctrl_addr(i);
    auto new_stub = CtrlService::NewStub(address.host() + ":" + std::to_string(address.port()));
    rpc_client_.AddStub(std::move(new_stub));
    rpc_client_.LoadServer(address.host(), rpc_client_.GetStubAt(i));
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
      for (size_t i = 0; i < rpc_client_.GetStubSize(); ++i) {
        grpc::ClientContext client_ctx;
        request.set_addr(this->process_ctx().ctrl_addr(i).host());
        GRPC_CHECK(rpc_client_.GetStubAt(i)->CallMethod<CtrlMethod::kLoadServer>(
            &client_ctx, request, &response))
            << "Machine " << i << " lost";
      }
      std::this_thread::sleep_for(std::chrono::seconds(sleep_second_dis(gen)));
    }
  });
}

void GrpcCtrlClient::Barrier(const std::string& barrier_name) { rpc_client_.Barrier(barrier_name); }

void GrpcCtrlClient::Barrier(const std::string& barrier_name, int32_t barrier_num) {
  rpc_client_.Barrier(barrier_name, barrier_num);
}

TryLockResult GrpcCtrlClient::TryLock(const std::string& name) { return rpc_client_.TryLock(name); }

void GrpcCtrlClient::NotifyDone(const std::string& name) { rpc_client_.NotifyDone(name); }

void GrpcCtrlClient::WaitUntilDone(const std::string& name) { rpc_client_.WaitUntilDone(name); }

void GrpcCtrlClient::PushKV(const std::string& k, const std::string& v) {
  rpc_client_.PushKV(k, v);
}

void GrpcCtrlClient::PushKV(const std::string& k, const PbMessage& msg) {
  rpc_client_.PushKV(k, msg);
}

void GrpcCtrlClient::PushKV(const std::string& k, std::function<void(std::string*)> VSetter) {
  rpc_client_.PushKV(k, VSetter);
}

void GrpcCtrlClient::PushMasterKV(const std::string& k, const PbMessage& msg) {
  rpc_client_.PushMasterKV(k, msg);
}

void GrpcCtrlClient::ClearKV(const std::string& k) { rpc_client_.ClearKV(k); }

void GrpcCtrlClient::ClearMasterKV(const std::string& k) { rpc_client_.ClearMasterKV(k); }

void GrpcCtrlClient::PullKV(const std::string& k, std::string* v) { rpc_client_.PullKV(k, v); }

void GrpcCtrlClient::PullKV(const std::string& k, PbMessage* msg) { rpc_client_.PullKV(k, msg); }

void GrpcCtrlClient::PullKV(const std::string& k, std::function<void(const std::string&)> VGetter) {
  rpc_client_.PullKV(k, VGetter);
}

void GrpcCtrlClient::PullMasterKV(const std::string& k, PbMessage* msg) {
  rpc_client_.PullMasterKV(k, msg);
}

void GrpcCtrlClient::Clear() { rpc_client_.Clear(); }

void GrpcCtrlClient::PushActEvent(const ActEvent& act_event) {
  rpc_client_.PushActEvent(act_event);
}

}  // namespace oneflow
