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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace {

static const int64_t machine_id_shl = 11 + 21 + 21;
static const int64_t thread_id_shl = 21 + 21;
static const int64_t local_work_stream_shl = 21;

EnvProto GetEnvProto() {
  EnvProto ret;
  for (size_t i = 0; i < 10; ++i) {
    auto* machine = ret.add_machine();
    machine->set_id(i);
    machine->set_addr("192.168.1." + std::to_string(i));
  }
  ret.set_ctrl_port(9527);
  return ret;
}

Resource GetResource() {
  Resource ret;
  ret.set_machine_num(10);
  ret.set_gpu_device_num(8);
  ret.set_cpu_device_num(5);
  ret.set_comm_net_worker_num(4);
  return ret;
}

void New() {
  Global<EnvDesc>::New(GetEnvProto());
  Global<NumProcessPerNode>::New()->set_value(1);
  Global<ResourceDesc, ForSession>::New(GetResource(), GlobalProcessCtx::NumOfProcessPerNode());
  Global<IDMgr>::New();
}

void Delete() {
  Global<IDMgr>::Delete();
  Global<NumProcessPerNode>::Delete();
  Global<ResourceDesc, ForSession>::Delete();
  Global<EnvDesc>::Delete();
}

}  // namespace

TEST(IDMgr, compile_regst_desc_id) {
  New();
  ASSERT_EQ(Global<IDMgr>::Get()->NewRegstDescId(), 0);
  ASSERT_EQ(Global<IDMgr>::Get()->NewRegstDescId(), 1);
  ASSERT_EQ(Global<IDMgr>::Get()->NewRegstDescId(), 2);
  Delete();
}

}  // namespace oneflow
