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
#ifndef ONEFLOW_CORE_CONTROL_RANK_INFO_BOOTSTRAP_SERVER_H_
#define ONEFLOW_CORE_CONTROL_RANK_INFO_BOOTSTRAP_SERVER_H_

#include "oneflow/core/control/bootstrap_server.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class RankInfoBootstrapServer final : public BootstrapServer {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RankInfoBootstrapServer);
  ~RankInfoBootstrapServer() override = default;

  RankInfoBootstrapServer(const BootstrapConf& bootstrap_conf);

  int64_t port() const { return port_; }
  Maybe<const std::vector<std::string>&> rank2host() const;

 private:
  void OnLoadServer(CtrlCall<CtrlMethod::kLoadServer>* call) override;

  int port_;
  const int64_t world_size_;
  // use std::shared_ptr as std::optional
  std::shared_ptr<std::vector<std::string>> rank2host_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_RANK_INFO_BOOTSTRAP_SERVER_H_
