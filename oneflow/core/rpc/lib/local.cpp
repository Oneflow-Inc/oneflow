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

#ifndef ONEFLOW_CORE_RPC_LIB_LOCAL_
#define ONEFLOW_CORE_RPC_LIB_LOCAL_
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/control/ctrl_client.h"

namespace oneflow {

CtrlClient::~CtrlClient() {
  {
    std::unique_lock<std::mutex> lck(need_heartbeat_thread_stop_mtx_);
    need_heartbeat_thread_stop_ = true;
  }
  heartbeat_thread_.join();
}

CtrlClient::CtrlClient(const ProcessCtx& process_ctx) : process_ctx_(process_ctx) {}

void RpcClient::Barrier(const std::string& barrier_name) {
  Barrier(barrier_name, Global<EnvDesc>::Get()->TotalMachineNum());
}

CtrlServer::CtrlServer(int ctrl_port) : RpcServer(), port_(ctrl_port) {}

CtrlServer::CtrlServer() : CtrlServer(0) {}

void CtrlServer::OnLoadServer(CtrlCall<CtrlMethod::kLoadServer>* call) { call->SendResponse(); }

void RpcClient::Barrier(const std::string& barrier_name, int32_t barrier_num) {}

TryLockResult RpcClient::TryLock(const std::string& name) {}

void RpcClient::NotifyDone(const std::string& name) {}

void RpcClient::WaitUntilDone(const std::string& name) {}

void RpcClient::PushKV(const std::string& k, std::function<void(std::string*)> VSetter) {}

void RpcClient::PushMasterKV(const std::string& k, std::function<void(std::string*)> VSetter) {}

void RpcClient::PushKV(const std::string& k, const std::string& v) {
  PushKV(k, [&](std::string* o) { *o = v; });
}

void RpcClient::PushKV(const std::string& k, const PbMessage& msg) {
  PushKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void RpcClient::PushMasterKV(const std::string& k, const PbMessage& msg) {
  PushMasterKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void RpcClient::ClearKV(const std::string& k) {}

void RpcClient::ClearMasterKV(const std::string& k) {}

void RpcClient::PullKV(const std::string& k, std::function<void(const std::string&)> VGetter) {}

void RpcClient::PullMasterKV(const std::string& k,
                             std::function<void(const std::string&)> VGetter) {}

void RpcClient::PullKV(const std::string& k, std::string* v) {
  PullKV(k, [&](const std::string& i) { *v = i; });
}

void RpcClient::PullKV(const std::string& k, PbMessage* msg) {
  PullKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void RpcClient::PullMasterKV(const std::string& k, PbMessage* msg) {
  PullMasterKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void RpcClient::PushActEvent(const ActEvent& act_event) {}

void RpcClient::Clear() {}

int32_t RpcClient::IncreaseCount(const std::string& k, int32_t v) {}

void RpcClient::EraseCount(const std::string& k) {}

RpcServer::~RpcServer() {}

void RpcServer::HandleRpcs() {}

void RpcServer::Init() {}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_LIB_LOCAL_
