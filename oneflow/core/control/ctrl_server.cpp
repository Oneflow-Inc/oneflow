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
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/job/profiler.h"
#include "oneflow/core/job/env_desc.h"
#include "grpc/grpc_posix.h"

namespace oneflow {

CtrlServer::CtrlServer() : RpcServer() {
  if (Global<EnvDesc>::Get()->has_bootstrap_conf_ctrl_port()) {
    port_ = Global<EnvDesc>::Get()->bootstrap_conf_ctrl_port();
  } else {
    port_ = 0;
  }
  Init();
  grpc::ServerBuilder server_builder;
  server_builder.SetMaxMessageSize(INT_MAX);
  int bound_port = 0;
  server_builder.AddListeningPort("0.0.0.0:" + std::to_string(port_),
                                  grpc::InsecureServerCredentials(), &bound_port);
  grpc_service_.reset(new CtrlService::AsyncService);
  server_builder.RegisterService(grpc_service_.get());
  cq_ = server_builder.AddCompletionQueue();
  grpc_server_ = server_builder.BuildAndStart();
  if (port() != 0) {
    CHECK_EQ(port(), bound_port) << "Port " << port() << " is unavailable";
  } else {
    port_ = bound_port;
    CHECK_NE(port(), 0);
  }
  LOG(INFO) << "CtrlServer listening on "
            << "0.0.0.0:" + std::to_string(port());
  loop_thread_ = std::thread(&CtrlServer::HandleRpcs, this);
}

void CtrlServer::OnLoadServer(CtrlCall<CtrlMethod::kLoadServer>* call) {
  call->SendResponse();
  EnqueueRequest<CtrlMethod::kLoadServer>();
}

}  // namespace oneflow
