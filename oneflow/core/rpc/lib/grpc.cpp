#ifdef RPC_BACKEND_GRPC

#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_bootstrap.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/env_desc.h"
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
  auto& env_desc = *Global<EnvDesc>::Get();
  if (env_desc.has_ctrl_bootstrap_conf()) {
    ctrl_bootstrap.reset(new RankInfoCtrlBootstrap(env_desc.bootstrap_conf()));
  } else {
    ctrl_bootstrap.reset(new HostListCtrlBootstrap(env_desc));
  }
  ctrl_bootstrap->InitProcessCtx(Global<CtrlServer>::Get()->port(), Global<ProcessCtx>::Get());
  return Maybe<void>::Ok();
}

Maybe<void> GrpcRpcManager::CreateServer() {
  Global<CtrlServer>::New(JUST(GetCtrlPort(*Global<EnvDesc>::Get())));
  return Maybe<void>::Ok();
}

Maybe<void> GrpcRpcManager::CreateClient() {
  auto* client = new GrpcCtrlClient(*Global<ProcessCtx>::Get());
  Global<CtrlClient>::SetAllocated(client);
  return Maybe<void>::Ok();
}

GrpcRpcManager::~GrpcRpcManager() {
  Global<CtrlClient>::Delete();
  Global<CtrlServer>::Delete();
}

}  // namespace oneflow

#endif  // RPC_BACKEND_GPRC
