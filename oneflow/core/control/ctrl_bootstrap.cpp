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
#include <map>
#include "oneflow/core/control/ctrl_bootstrap.h"
#include "oneflow/core/control/host_list_bootstrap_server.h"
#include "oneflow/core/control/host_list_bootstrap_client.h"
#include "oneflow/core/control/rank_info_bootstrap_server.h"
#include "oneflow/core/control/rank_info_bootstrap_client.h"

namespace oneflow {

Maybe<void> CtrlBootstrap::InitProcessCtx(int64_t port, ProcessCtx* ret_process_ctx) {
  std::vector<ProcessCtx> rank2process_ctx;
  if (rank() == 0) {
    ProcessCtx process_ctx;
    {
      process_ctx.set_rank(rank());
      Address* addr = process_ctx.mutable_ctrl_addr()->Add();
      JUST(SetCurrentHostByMaster(addr));
      addr->set_port(port);
    }
    rank2process_ctx.push_back(process_ctx);
    for (int64_t world_rank = 1; world_rank < world_size(); ++world_rank) {
      std::string key = std::string("GetProcessCtx") + std::to_string(world_rank);
      ProcessCtx cur_process_ctx;
      mut_bootstrap_client()->PullMasterKV(key, &cur_process_ctx);
      CHECK_EQ_OR_RETURN(world_rank, rank2process_ctx.size());
      CHECK_EQ_OR_RETURN(world_rank, cur_process_ctx.rank());
      rank2process_ctx.push_back(cur_process_ctx);
    }
  } else {
    std::string key = std::string("GetProcessCtx") + std::to_string(rank());
    ProcessCtx cur_process_ctx;
    {
      cur_process_ctx.set_rank(rank());
      Address* addr = cur_process_ctx.mutable_ctrl_addr()->Add();
      JUST(SetCurrentHostByWorker(addr));
      addr->set_port(port);
    }
    mut_bootstrap_client()->PushMasterKV(key, cur_process_ctx);
  }

  mut_bootstrap_client()->Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));

  if (rank() == 0) {
    ret_process_ctx->set_rank(rank());
    ret_process_ctx->mutable_ctrl_addr()->Clear();
    for (const auto& process_ctx : rank2process_ctx) {
      CHECK_EQ_OR_RETURN(process_ctx.ctrl_addr_size(), 1);
      Address* addr = ret_process_ctx->mutable_ctrl_addr()->Add();
      *addr = process_ctx.ctrl_addr(0);
      if (!addr->has_host()) { JUST(SetHostByMaster(addr, process_ctx.rank())); }
      CHECK_OR_RETURN(addr->has_host());
    }
    mut_bootstrap_client()->PushMasterKV("BroadcastProcessCtx", *ret_process_ctx);
  } else {
    mut_bootstrap_client()->PullMasterKV("BroadcastProcessCtx", ret_process_ctx);
    ret_process_ctx->set_rank(rank());
  }

  mut_bootstrap_client()->Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));

  LOG(INFO) << "\n" << ret_process_ctx->DebugString();
  return Maybe<void>::Ok();
}

HostListCtrlBootstrap::HostListCtrlBootstrap(const EnvDesc& env_desc) : CtrlBootstrap() {
  bootstrap_server_.reset(new HostListBootstrapServer(env_desc));
  bootstrap_client_.reset(new HostListBootstrapClient(env_desc));
  bootstrap_client_->Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));
  host_ = bootstrap_server_->this_machine_addr();
  rank_ = env_desc.GetMachineId(host_);
  world_size_ = env_desc.TotalMachineNum();
}

HostListCtrlBootstrap::~HostListCtrlBootstrap() {
  bootstrap_client_.reset();
  bootstrap_server_.reset();
}

Maybe<void> HostListCtrlBootstrap::SetHostByMaster(Address* addr, int64_t world_rank) const {
  return Maybe<void>::Ok();
}
Maybe<void> HostListCtrlBootstrap::SetCurrentHostByMaster(Address* addr) const {
  addr->set_host(host());
  return Maybe<void>::Ok();
}
Maybe<void> HostListCtrlBootstrap::SetCurrentHostByWorker(Address* addr) const {
  addr->set_host(host());
  return Maybe<void>::Ok();
}

BootstrapServer* HostListCtrlBootstrap::mut_bootstrap_server() {
  return bootstrap_server_.get();
}
BootstrapClient* HostListCtrlBootstrap::mut_bootstrap_client() {
  return bootstrap_client_.get();
}

RankInfoCtrlBootstrap::RankInfoCtrlBootstrap(const BootstrapConf& bootstrap_conf)
    : CtrlBootstrap(), bootstrap_conf_(bootstrap_conf) {
  bootstrap_server_.reset(new RankInfoBootstrapServer(bootstrap_conf));
  bootstrap_client_.reset(new RankInfoBootstrapClient(bootstrap_conf));
  bootstrap_client_->Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));
  master_host_ = bootstrap_conf.master_addr().host();
  rank_ = bootstrap_conf.rank();
  world_size_ = bootstrap_conf.world_size();
}

RankInfoCtrlBootstrap::~RankInfoCtrlBootstrap() {
  bootstrap_client_.reset();
  bootstrap_server_.reset();
}

Maybe<void> RankInfoCtrlBootstrap::SetHostByMaster(Address* addr, int64_t world_rank) const {
  const auto& rank2host = JUST(bootstrap_server_->rank2host());
  CHECK_EQ_OR_RETURN(rank2host.size(), world_size());
  CHECK_GE_OR_RETURN(world_rank, 0);
  CHECK_LT_OR_RETURN(world_rank, rank2host.size());
  addr->set_host(rank2host.at(world_rank));
  return Maybe<void>::Ok();
}

Maybe<void> RankInfoCtrlBootstrap::SetCurrentHostByMaster(Address* addr) const {
  CHECK_EQ_OR_RETURN(rank(), 0);
  if (bootstrap_conf_.has_host()) {
    CHECK(bootstrap_conf_.host().has_host());
    addr->set_host(bootstrap_conf_.host().host());
  } else {
    addr->set_host(master_host_);
  }
  return Maybe<void>::Ok();
}

Maybe<void> RankInfoCtrlBootstrap::SetCurrentHostByWorker(Address* addr) const {
  CHECK_NE_OR_RETURN(rank(), 0);
  if (host() != "") { addr->set_host(host()); }
  return Maybe<void>::Ok();
}

BootstrapServer* RankInfoCtrlBootstrap::mut_bootstrap_server() {
  return bootstrap_server_.get();
}
BootstrapClient* RankInfoCtrlBootstrap::mut_bootstrap_client() {
  return bootstrap_client_.get();
}

}  // namespace oneflow
