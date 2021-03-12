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
#include "glog/logging.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/rpc/include/local/ctrl.h"

namespace oneflow {

CtrlClient::~CtrlClient() {}

CtrlClient::CtrlClient(const ProcessCtx& process_ctx) : process_ctx_(process_ctx) {
  CHECK(process_ctx.ctrl_addr_size() == 0);
  CHECK(process_ctx.node_size() == 0);
}

CtrlServer::CtrlServer(int /* ctrl_port */) : RpcServer() {}

CtrlServer::CtrlServer() : CtrlServer(0) {}

void RpcClient::Barrier(const std::string& barrier_name) {
  Barrier(barrier_name, Global<EnvDesc>::Get()->TotalMachineNum());
}

void CtrlServer::OnLoadServer(CtrlCall<CtrlMethod::kLoadServer>* call) { call->SendResponse(); }

void RpcClient::Barrier(const std::string& barrier_name, int32_t barrier_num) {
  CHECK(barrier_num == 1);
}

TryLockResult RpcClient::TryLock(const std::string& name) { UNIMPLEMENTED(); }

void RpcClient::NotifyDone(const std::string& name) { UNIMPLEMENTED(); }

void RpcClient::WaitUntilDone(const std::string& name) { UNIMPLEMENTED(); }

void RpcClient::PushKV(const std::string& k, std::function<void(std::string*)> VSetter) {
  VSetter(&kv_[k]);
}

void RpcClient::PushMasterKV(const std::string& k, std::function<void(std::string*)> VSetter) {
  PushKV(k, VSetter);
}

void RpcClient::PushKV(const std::string& k, const std::string& v) {
  PushKV(k, [&](std::string* o) { *o = v; });
}

void RpcClient::PushKV(const std::string& k, const PbMessage& msg) {
  PushKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void RpcClient::PushMasterKV(const std::string& k, const PbMessage& msg) {
  PushMasterKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void RpcClient::ClearKV(const std::string& k) { kv_.erase(k); }

void RpcClient::ClearMasterKV(const std::string& k) { ClearKV(k); }

void RpcClient::PullKV(const std::string& k, std::function<void(const std::string&)> VGetter) {
  VGetter(kv_.at(k));
}

void RpcClient::PullMasterKV(const std::string& k,
                             std::function<void(const std::string&)> VGetter) {
  PullKV(k, VGetter);
}

void RpcClient::PullKV(const std::string& k, std::string* v) {
  PullKV(k, [&](const std::string& i) { *v = i; });
}

void RpcClient::PullKV(const std::string& k, PbMessage* msg) {
  PullKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void RpcClient::PullMasterKV(const std::string& k, PbMessage* msg) {
  PullMasterKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void RpcClient::Clear() { UNIMPLEMENTED(); }

int32_t RpcClient::IncreaseCount(const std::string& k, int32_t v) { UNIMPLEMENTED(); }

void RpcClient::EraseCount(const std::string& k) { UNIMPLEMENTED(); }

RpcServer::~RpcServer() {}

void RpcServer::HandleRpcs() { UNIMPLEMENTED(); }

void RpcServer::Init() { UNIMPLEMENTED(); }

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_LIB_LOCAL_
