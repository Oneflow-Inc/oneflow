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

Maybe<void> InitCtrlConfFromEnvDesc(const EnvDesc& env_desc, CtrlConf* ret_ctrl_conf) {
  HostListBootstrapServer bootstrap_server(env_desc);
  HostListBootstrapClient bootstrap_client(env_desc);
  bootstrap_client.Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));
  int64_t this_machine_id = env_desc.GetMachineId(bootstrap_server.this_machine_addr());
  std::vector<CtrlConf> rank2ctrl_conf;
  if (this_machine_id == 0) {
    CtrlConf ctrl_conf;
    {
      Address* addr = ctrl_conf.mutable_ctrl_addr()->Add();
      addr->set_host(bootstrap_server.this_machine_addr());
      addr->set_port(env_desc.ctrl_port());
      ctrl_conf.set_rank(this_machine_id);
      rank2ctrl_conf.push_back(ctrl_conf);
    }
    for (int64_t machine_id = 1; machine_id < env_desc.TotalMachineNum(); ++machine_id) {
      std::string key = std::string("GetCtrlConf") + std::to_string(machine_id);
      CtrlConf cur_ctrl_conf;
      bootstrap_client.PullMasterKV(key, &cur_ctrl_conf);
      CHECK_EQ_OR_RETURN(machine_id, rank2ctrl_conf.size());
      CHECK_EQ_OR_RETURN(machine_id, cur_ctrl_conf.rank());
      rank2ctrl_conf.push_back(cur_ctrl_conf);
    }
  } else {
    std::string key = std::string("GetCtrlConf") + std::to_string(this_machine_id);
    CtrlConf cur_ctrl_conf;
    cur_ctrl_conf.set_rank(this_machine_id);
    Address* addr = cur_ctrl_conf.mutable_ctrl_addr()->Add();
    addr->set_host(bootstrap_server.this_machine_addr());
    addr->set_port(env_desc.ctrl_port());
    bootstrap_client.PushMasterKV(key, cur_ctrl_conf);
  }

  bootstrap_client.Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));

  if (this_machine_id == 0) {
    ret_ctrl_conf->set_rank(this_machine_id);
    for (const auto& ctrl_conf : rank2ctrl_conf) {
      CHECK_EQ_OR_RETURN(ctrl_conf.ctrl_addr_size(), 1);
      *ret_ctrl_conf->mutable_ctrl_addr()->Add() = ctrl_conf.ctrl_addr(0);
    }
    for (int64_t machine_id = 1; machine_id < env_desc.TotalMachineNum(); ++machine_id) {
      std::string key = std::string("BroadcastCtrlConf") + std::to_string(machine_id);
      bootstrap_client.PushMasterKV(key, *ret_ctrl_conf);
    }
  } else {
    std::string key = std::string("BroadcastCtrlConf") + std::to_string(this_machine_id);
    bootstrap_client.PullMasterKV(key, ret_ctrl_conf);
    ret_ctrl_conf->set_rank(this_machine_id);
  }

  bootstrap_client.Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));

  LOG(ERROR) << "\n" << ret_ctrl_conf->DebugString();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
