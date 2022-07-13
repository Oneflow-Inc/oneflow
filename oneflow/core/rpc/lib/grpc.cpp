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
#ifdef RPC_BACKEND_GRPC

#include "oneflow/core/control/ctrl_bootstrap.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/rpc/include/grpc.h"

namespace oneflow {

namespace {

Maybe<int> GetCtrlPort(const EnvDesc& env_desc) {
  int port = 0;
  if (env_desc.has_bootstrap_conf_ctrl_port()) { port = env_desc.bootstrap_conf_ctrl_port(); }
  return port;
}

}  // namespace

Maybe<void> GrpcRpcManager::Bootstrap() {
  std::shared_ptr<CtrlBootstrap> ctrl_bootstrap;
  auto& env_desc = *Singleton<EnvDesc>::Get();
  if (env_desc.has_ctrl_bootstrap_conf()) {
    ctrl_bootstrap.reset(new RankInfoCtrlBootstrap(env_desc.bootstrap_conf()));
  } else {
    ctrl_bootstrap.reset(new HostListCtrlBootstrap(env_desc));
  }
  JUST(ctrl_bootstrap->InitProcessCtx(Singleton<CtrlServer>::Get()->port(),
                                      Singleton<ProcessCtx>::Get()));
  return Maybe<void>::Ok();
}

Maybe<void> GrpcRpcManager::CreateServer() {
  Singleton<CtrlServer>::New(JUST(GetCtrlPort(*Singleton<EnvDesc>::Get())));
  return Maybe<void>::Ok();
}

Maybe<void> GrpcRpcManager::CreateClient() {
  auto* client = new GrpcCtrlClient(*Singleton<ProcessCtx>::Get());
  Singleton<CtrlClient>::SetAllocated(client);
  return Maybe<void>::Ok();
}

GrpcRpcManager::~GrpcRpcManager() {
  auto* grpc_client = dynamic_cast<GrpcCtrlClient*>(Singleton<CtrlClient>::Get());
  CHECK_NOTNULL(grpc_client);
  grpc_client->StopHeartbeat();
  OF_ENV_BARRIER();
  Singleton<CtrlClient>::Delete();
  CHECK_NOTNULL(Singleton<CtrlServer>::Get());
  Singleton<CtrlServer>::Delete();
}

}  // namespace oneflow

#endif  // RPC_BACKEND_GRPC
