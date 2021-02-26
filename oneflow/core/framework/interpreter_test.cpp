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
#include "oneflow/core/framework/interpreter.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/virtual_machine_scope.h"

namespace oneflow {
namespace test {

namespace {

class TestVirtualMachineScope {
 public:
  TestVirtualMachineScope(int64_t gpu_device_num, int64_t cpu_device_num) {
    Global<ProcessCtx>::New();
    Global<ProcessCtx>::Get()->set_rank(0);
    test_resource_desc_scope_.reset(new vm::TestResourceDescScope(gpu_device_num, cpu_device_num));
    virtual_machine_scope_.reset(
        new vm::VirtualMachineScope(Global<ResourceDesc, ForSession>::Get()->resource()));
  }

  ~TestVirtualMachineScope() {
    virtual_machine_scope_.reset();
    test_resource_desc_scope_.reset();
    Global<ProcessCtx>::Delete();
  }

 private:
  std::unique_ptr<vm::TestResourceDescScope> test_resource_desc_scope_;
  std::unique_ptr<vm::VirtualMachineScope> virtual_machine_scope_;
};

TEST(Interpreter, do_nothing) {
  TestVirtualMachineScope vm_scope(0, 1);
  CHECK_JUST(PhysicalInterpreter().Run(
      [](InstructionsBuilder* builder) -> Maybe<void> { return Maybe<void>::Ok(); }));
}

TEST(Interpreter, new_scope) {
  TestVirtualMachineScope vm_scope(0, 1);
  int64_t symbol_id = 0;
  CHECK_JUST(PhysicalInterpreter().Run([&](InstructionsBuilder* builder) -> Maybe<void> {
    int64_t parallel_desc_symbol_id = 0;
    {
      cfg::ParallelConf parallel_conf;
      parallel_conf.set_device_tag("cpu");
      parallel_conf.add_device_name("0:0");
      parallel_desc_symbol_id =
          JUST(builder->FindOrCreateSymbolId<cfg::ParallelConf>(parallel_conf));
    }
    int64_t job_desc_symbol_id = 0;
    {
      cfg::JobConfigProto job_conf;
      job_conf.mutable_predict_conf();
      job_desc_symbol_id = JUST(builder->FindOrCreateSymbolId<cfg::JobConfigProto>(job_conf));
    }
    cfg::ScopeProto scope_proto;
    scope_proto.set_job_desc_symbol_id(job_desc_symbol_id);
    scope_proto.set_device_parallel_desc_symbol_id(parallel_desc_symbol_id);
    scope_proto.set_host_parallel_desc_symbol_id(parallel_desc_symbol_id);
    scope_proto.mutable_opt_mirrored_parallel_conf();
    scope_proto.set_session_id(1);
    symbol_id = JUST(builder->FindOrCreateSymbolId<cfg::ScopeProto>(scope_proto));
    return Maybe<void>::Ok();
  }));
  ASSERT_NE(symbol_id, 0);
}

}  // namespace

}  // namespace test
}  // namespace oneflow
