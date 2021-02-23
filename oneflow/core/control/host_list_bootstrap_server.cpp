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
#include "oneflow/core/control/host_list_bootstrap_server.h"
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/job/profiler.h"
#include "grpc/grpc_posix.h"

namespace oneflow {

HostListBootstrapServer::HostListBootstrapServer(const EnvDesc& env_desc)
    : BootstrapServer(), is_first_connect_(true), this_machine_addr_("") {
  Init();
  int port = env_desc.ctrl_port();
  grpc::ServerBuilder server_builder;
  server_builder.SetMaxMessageSize(INT_MAX);
  int bound_port = 0;
  server_builder.AddListeningPort("0.0.0.0:" + std::to_string(port),
                                  grpc::InsecureServerCredentials(), &bound_port);
  grpc_service_.reset(new CtrlService::AsyncService);
  server_builder.RegisterService(grpc_service_.get());
  cq_ = server_builder.AddCompletionQueue();
  grpc_server_ = server_builder.BuildAndStart();
  CHECK_EQ(port, bound_port) << "Port " << port << " is unavailable";
  LOG(INFO) << "HostListBootstrapServer listening on "
            << "0.0.0.0:" + std::to_string(port);
  loop_thread_ = std::thread(&HostListBootstrapServer::HandleRpcs, this);
}

void HostListBootstrapServer::OnLoadServer(CtrlCall<CtrlMethod::kLoadServer>* call) {
  if (this->is_first_connect_) {
    this->this_machine_addr_ = call->request().addr();
    this->is_first_connect_ = false;
  } else {
    CHECK_EQ(call->request().addr(), this->this_machine_addr_);
  }
  call->SendResponse();
  EnqueueRequest<CtrlMethod::kLoadServer>();
}

}  // namespace oneflow
