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
#ifndef ONEFLOW_CORE_CONTROL_CTRL_BOOTSTRAP_H_
#define ONEFLOW_CORE_CONTROL_CTRL_BOOTSTRAP_H_

#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class ProcessCtx;
class WorkerProcessInfo;
class BootstrapServer;
class BootstrapClient;

class CtrlBootstrap {
 public:
  virtual ~CtrlBootstrap() {}

  Maybe<void> InitProcessCtx(int64_t port, ProcessCtx* process_ctx);

 protected:
  virtual int64_t rank() const = 0;
  virtual int64_t world_size() const = 0;
  virtual Maybe<void> SetHostByMaster(Address*, int64_t world_rank) const = 0;
  virtual Maybe<void> SetCurrentHostByMaster(WorkerProcessInfo*) const = 0;
  virtual Maybe<void> SetCurrentHostByWorker(WorkerProcessInfo*) const = 0;

  virtual BootstrapServer* mut_bootstrap_server() = 0;
  virtual BootstrapClient* mut_bootstrap_client() = 0;

  CtrlBootstrap() = default;
};

class HostListBootstrapServer;
class HostListBootstrapClient;

class HostListCtrlBootstrap final : public CtrlBootstrap {
 public:
  explicit HostListCtrlBootstrap(const EnvDesc& env_desc);
  ~HostListCtrlBootstrap() override;

 private:
  int64_t rank() const override { return rank_; }
  int64_t world_size() const override { return world_size_; }

  std::string host() const { return host_; }

  Maybe<void> SetHostByMaster(Address*, int64_t world_rank) const override;
  Maybe<void> SetCurrentHostByMaster(WorkerProcessInfo*) const override;
  Maybe<void> SetCurrentHostByWorker(WorkerProcessInfo*) const override;

  BootstrapServer* mut_bootstrap_server() override;
  BootstrapClient* mut_bootstrap_client() override;

  // Uses shared_ptr and forward declaration to avoid `#include ...`
  std::shared_ptr<HostListBootstrapServer> bootstrap_server_;
  std::shared_ptr<HostListBootstrapClient> bootstrap_client_;

  std::string host_;
  int64_t rank_;
  int64_t world_size_;
};

class RankInfoBootstrapServer;
class RankInfoBootstrapClient;

class RankInfoCtrlBootstrap final : public CtrlBootstrap {
 public:
  explicit RankInfoCtrlBootstrap(const BootstrapConf& bootstrap_conf);
  ~RankInfoCtrlBootstrap() override;

 private:
  int64_t rank() const override { return rank_; }
  int64_t world_size() const override { return world_size_; }

  Maybe<void> SetHostByMaster(Address*, int64_t world_rank) const override;
  Maybe<void> SetCurrentHostByMaster(WorkerProcessInfo*) const override;
  Maybe<void> SetCurrentHostByWorker(WorkerProcessInfo*) const override;

  BootstrapServer* mut_bootstrap_server() override;
  BootstrapClient* mut_bootstrap_client() override;

  // Uses shared_ptr and forward declaration to avoid `#include ...`
  std::shared_ptr<RankInfoBootstrapServer> bootstrap_server_;
  std::shared_ptr<RankInfoBootstrapClient> bootstrap_client_;

  std::string master_host_;
  BootstrapConf bootstrap_conf_;
  int64_t rank_;
  int64_t world_size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_BOOTSTRAP_H_
