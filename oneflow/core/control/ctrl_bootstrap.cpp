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

namespace oneflow {

Maybe<void> CtrlBootstrap::InitProcessCtx(int64_t port, ProcessCtx* ret_process_ctx,
    const std::function<Maybe<void>(Address*, int64_t world_rank)>& SetHostByMaster,
    const std::function<void(Address*)>& SetCurrentHostByMaster,
    const std::function<void(Address*)>& SetCurrentHostByWorker) {
  std::vector<ProcessCtx> rank2process_ctx;
  if (rank() == 0) {
    ProcessCtx process_ctx;
    {
      process_ctx.set_rank(rank());
      Address* addr = process_ctx.mutable_ctrl_addr()->Add();
      SetCurrentHostByMaster(addr);
      addr->set_port(port);
    }
    rank2process_ctx.push_back(process_ctx);
    for (int64_t world_rank = 1; world_rank < world_size(); ++world_rank) {
      std::string key = std::string("GetProcessCtx") + std::to_string(world_rank);
      ProcessCtx cur_process_ctx;
      bootstrap_client_->PullMasterKV(key, &cur_process_ctx);
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
      SetCurrentHostByWorker(addr);
      addr->set_port(port);
    }
    bootstrap_client_->PushMasterKV(key, cur_process_ctx);
  }

  bootstrap_client_->Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));

  if (rank() == 0) {
    ret_process_ctx->set_rank(rank());
    ret_process_ctx->mutable_ctrl_addr()->Clear();
    for (const auto& process_ctx : rank2process_ctx) {
      CHECK_EQ_OR_RETURN(process_ctx.ctrl_addr_size(), 1);
      Address* addr = ret_process_ctx->mutable_ctrl_addr()->Add();
      *addr = process_ctx.ctrl_addr(0);
      JUST(SetHostByMaster(addr, process_ctx.rank()));
    }
    bootstrap_client_->PushMasterKV("BroadcastProcessCtx", *ret_process_ctx);
  } else {
    bootstrap_client_->PullMasterKV("BroadcastProcessCtx", ret_process_ctx);
    ret_process_ctx->set_rank(rank());
  }

  bootstrap_client_->Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));

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

Maybe<void> HostListCtrlBootstrap::InitProcessCtx(int64_t port, ProcessCtx* ret_process_ctx) {
  const auto& SetHostByMaster = [&](Address*, int64_t) { return Maybe<void>::Ok(); };
  const auto& SetCurrentHostByMaster = [&](Address* addr) { addr->set_host(host()); };
  const auto& SetCurrentHostByWorker = [&](Address* addr) { addr->set_host(host()); };
  return InitProcessCtx(port, ret_process_ctx, SetHostByMaster, SetCurrentHostByMaster, SetCurrentHostByWorker);
}

RankInfoCtrlBootstrap::RankInfoCtrlBootstrap(const BootstrapConf& bootstrap_conf) : CtrlBootstrap() {
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

Maybe<void> RankInfoCtrlBootstrap::InitProcessCtx(int64_t port, ProcessCtx* ret_process_ctx) {
  const auto& SetHostByMaster = [&](Address* addr, int64_t world_rank) {
    const auto& rank2host = JUST(bootstrap_server_.rank2host());
    CHECK_EQ_OR_RETURN(rank2host, world_size());
    CHECK_GT_OR_RETURN(world_rank, 0);
    CHECK_LT_OR_RETURN(world_rank, rank2host.size());
    addr->set_host(rank2host.at(world_rank));
    return Maybe<void>::Ok();
  };
  const auto& SetCurrentHostByMaster = [&](Address* addr) {
    CHECK_EQ(rank(), 0);
    addr->set_host(master_host_);
  };
  const auto& SetCurrentHostByWorker = [&](Address* addr) {
    CHECK_NE(rank(), 0);
    // do nothing
  };
  return InitProcessCtx(port, ret_process_ctx, SetHostByMaster, SetCurrentHostByMaster, SetCurrentHostByWorker);
}


}  // namespace oneflow
