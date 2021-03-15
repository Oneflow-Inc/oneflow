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
#include "oneflow/core/job/env.pb.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_bootstrap.h"
#include "oneflow/core/control/ctrl_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

#ifdef OF_PLATFORM_POSIX

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>

namespace oneflow {

namespace {

EnvProto GetEnvProto(int port) {
  EnvProto ret;
  auto* machine0 = ret.add_machine();
  machine0->set_id(0);
  machine0->set_addr("127.0.0.1");
  ret.set_ctrl_port(port);
  return ret;
}

Resource GetResource() {
  Resource ret;
  ret.set_machine_num(1);
  ret.set_gpu_device_num(0);
  ret.set_cpu_device_num(1);
  ret.set_comm_net_worker_num(1);
  return ret;
}

}  // namespace

TEST(CtrlServer, new_delete) {
  int port = CtrlUtil().FindAvailablePort();
  if (port == -1) { return; }
  EnvProto env_proto = GetEnvProto(port);
  Global<EnvDesc>::New(env_proto);
  Global<CtrlServer>::New();
  Global<ProcessCtx>::New();
  CHECK_JUST(HostListCtrlBootstrap(*Global<EnvDesc>::Get())
                 .InitProcessCtx(Global<CtrlServer>::Get()->port(), Global<ProcessCtx>::Get()));
  Global<CtrlClient>::New(*Global<ProcessCtx>::Get());
  Global<ResourceDesc, ForEnv>::New(GetResource());
  Global<ResourceDesc, ForSession>::New(GetResource());

  // do test
  // OF_ENV_BARRIER();

  Global<ResourceDesc, ForSession>::Delete();
  Global<ResourceDesc, ForEnv>::Delete();
  Global<CtrlClient>::Delete();
  Global<ProcessCtx>::Delete();
  Global<CtrlServer>::Delete();
  Global<EnvDesc>::Delete();
}

}  // namespace oneflow

#endif  // OF_PLATFORM_POSIX
