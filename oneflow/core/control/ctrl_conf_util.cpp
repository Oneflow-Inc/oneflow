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
#include "oneflow/core/control/ctrl_conf_util.h"
#include "oneflow/core/control/host_list_boot_strap_server.h"
#include "oneflow/core/control/host_list_boot_strap_client.h"

namespace oneflow {

void InitConfFromEnvDesc(const EnvDesc& env_desc) {
  std::shared_ptr<HostListBootStrapServer> host_list_boot_strap_server =
      std::make_shared<HostListBootStrapServer>(env_desc);
  std::shared_ptr<HostListBootStrapClient> host_list_boot_strap_client =
      std::make_shared<HostListBootStrapClient>(env_desc);

  int64_t this_machine_id = env_desc.GetMachineId(host_list_boot_strap_server->this_machine_addr());
  std::map<int64_t, std::shared_ptr<CtrlConf>> rank2ctrl_conf;

  if (this_machine_id == 0) {
    std::shared_ptr<CtrlConf> ctrl_conf = std::make_shared<CtrlConf>();
    Address addr;
    addr.set_host(host_list_boot_strap_server->this_machine_addr());
    addr.set_port(env_desc.ctrl_port());
    *(ctrl_conf->mutable_ctrl_addrs()->Add()) = addr;
    ctrl_conf->set_rank(this_machine_id);
    rank2ctrl_conf.emplace(ctrl_conf->rank(), ctrl_conf);
    for (int64_t machine_id = 1; machine_id < env_desc.TotalMachineNum(); ++machine_id) {
      std::string key = std::string("InitCtrlConf") + std::to_string(machine_id);
      std::shared_ptr<CtrlConf> ctrl_conf = std::make_shared<CtrlConf>();
      host_list_boot_strap_client->PullMasterKV(key, ctrl_conf.get());
      rank2ctrl_conf.emplace(ctrl_conf->rank(), ctrl_conf);
    }
  } else {
    std::string key = std::string("InitCtrlConf") + std::to_string(this_machine_id);
    std::shared_ptr<CtrlConf> ctrl_conf = std::make_shared<CtrlConf>();
    Address addr;
    addr.set_host(host_list_boot_strap_server->this_machine_addr());
    addr.set_port(env_desc.ctrl_port());
    *(ctrl_conf->mutable_ctrl_addrs()->Add()) = addr;
    ctrl_conf->set_rank(this_machine_id);
    host_list_boot_strap_client->PushMasterKV(key, *ctrl_conf);
  }

  host_list_boot_strap_client->Barrier(__FILE__ ":" OF_PP_STRINGIZE(__LINE__));

  Global<CtrlConf>::New();
  Global<CtrlConf>::Get()->set_rank(this_machine_id);
  for (const auto& pair : rank2ctrl_conf) {
    Global<CtrlConf>::Get()->mutable_ctrl_addrs()->MergeFrom(pair.second->ctrl_addrs());
  }

  if (this_machine_id == 0) {
    for (int64_t machine_id = 1; machine_id < env_desc.TotalMachineNum(); ++machine_id) {
      std::string key = std::string("BroadcastCtrlConf") + std::to_string(machine_id);
      host_list_boot_strap_client->PushMasterKV(key, *Global<CtrlConf>::Get());
    }
  } else {
    std::string key = std::string("BroadcastCtrlConf") + std::to_string(this_machine_id);
    host_list_boot_strap_client->PullMasterKV(key, Global<CtrlConf>::Get());
    Global<CtrlConf>::Get()->set_rank(this_machine_id);
  }

  host_list_boot_strap_client.reset();
  host_list_boot_strap_server.reset();
}

}  // namespace oneflow
