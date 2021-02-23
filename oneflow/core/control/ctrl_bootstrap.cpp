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

HostListCtrlBootstrap::~HostListCtrlBootstrap() {
  bootstrap_client_.reset();
  bootstrap_server_.reset();
}

HostListCtrlBootstrap::HostListCtrlBootstrap(const EnvDesc& env_desc) : CtrlBootstrap() {
  bootstrap_server_.reset(new HostListBootstrapServer(env_desc));
  bootstrap_client_.reset(new HostListBootstrapClient(env_desc));
  bootstrap_client_->Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));
  host_ = bootstrap_server_->this_machine_addr();
  rank_ = env_desc.GetMachineId(host_);
  world_size_ = env_desc.TotalMachineNum();
}

Maybe<void> HostListCtrlBootstrap::InitProcessCtx(int64_t port, ProcessCtx* ret_process_ctx) {
  std::vector<ProcessCtx> rank2process_ctx;
  if (rank() == 0) {
    ProcessCtx process_ctx;
    {
      process_ctx.set_rank(rank());
      Address* addr = process_ctx.mutable_ctrl_addr()->Add();
      addr->set_host(host());
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
      addr->set_host(host());
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
      *ret_process_ctx->mutable_ctrl_addr()->Add() = process_ctx.ctrl_addr(0);
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

}  // namespace oneflow
