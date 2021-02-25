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
#ifndef ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_
#define ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_

#include "oneflow/core/control/rpc_server.h"

namespace oneflow {

class CtrlServer final : public RpcServer {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlServer);
  ~CtrlServer() override {}

  CtrlServer();
  // port may be configured in bootstrap_conf
  CtrlServer(int ctrl_port);

  int64_t port() const { return port_; }

 private:
  void OnLoadServer(CtrlCall<CtrlMethod::kLoadServer>* call) override;
  int port_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_
