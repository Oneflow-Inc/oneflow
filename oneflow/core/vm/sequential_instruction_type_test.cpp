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
// include sstream first to avoid some compiling error
// caused by the following trick
// reference: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65899
#include <sstream>
#include <thread>
#include <typeinfo>
#define private public
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/vm_desc.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/no_arg_cb_phy_instr_operand.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

struct GlobalProcessCtxScope {
  GlobalProcessCtxScope() {
    auto* ctx = Global<ProcessCtx>::New();
    ctx->mutable_ctrl_addr()->Add();
    ctx->set_rank(0);
    ctx->set_node_size(1);
  }
  ~GlobalProcessCtxScope() { Global<ProcessCtx>::Delete(); }
};

TEST(SequentialInstruction, front_seq_compute) {
  GlobalProcessCtxScope scope;
  auto vm_desc = intrusive::make_shared<VmDesc>(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(),
                                      {"NewObject", "ComputeRankFrontSeqCallback"});
  auto vm = intrusive::make_shared<VirtualMachine>(vm_desc.Get());
  InstructionMsgList list;
  {
    int64_t logical_object_id = TestUtil::NewObject(&list, "cpu", "0:0");
    list.EmplaceBack(NewInstruction("DeleteObject")->add_del_operand(logical_object_id));
    ASSERT_TRUE(vm->pending_msg_list().empty());
  }
  int64_t sixsixsix = 0;
  {
    auto instruction = NewInstruction("ComputeRankFrontSeqCallback");
    instruction->add_int64_operand(GlobalProcessCtx::Rank());
    const auto Callback = [&]() { sixsixsix = 666; };
    *instruction->mut_phy_instr_operand() = std::make_shared<vm::NoArgCbPhyInstrOperand>(Callback);
    list.EmplaceBack(std::move(instruction));
  }
  bool compute_finished = false;
  bool is_666 = false;
  {
    auto instruction = NewInstruction("CtrlComputeRankFrontSeqCallback");
    instruction->add_int64_operand(GlobalProcessCtx::Rank());
    const auto Callback = [&]() {
      is_666 = sixsixsix == 666;
      compute_finished = true;
    };
    *instruction->mut_phy_instr_operand() = std::make_shared<vm::NoArgCbPhyInstrOperand>(Callback);
    list.EmplaceBack(std::move(instruction));
  }
  BlockingCounter bc(1);
  std::thread t([&]() {
    while (!compute_finished) {
      vm->Schedule();
      INTRUSIVE_FOR_EACH_PTR(t, vm->mut_thread_ctx_list()) { t->TryReceiveAndRun(); }
    }
    bc.Decrease();
  });
  CHECK_JUST(vm->Receive(&list));
  bc.WaitUntilCntEqualZero();
  ASSERT_TRUE(is_666);
  ASSERT_TRUE(vm->Empty());
  t.join();
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
