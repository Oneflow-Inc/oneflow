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
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/no_arg_cb_phy_instr_operand.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"

namespace oneflow {
namespace vm {

namespace test {

namespace {

struct GlobalProcessCtxScope {
  GlobalProcessCtxScope() {
    auto* ctx = Global<ProcessCtx>::New();
    ctx->set_rank(0);
    ctx->set_node_size(1);
    Global<NumProcessPerNode>::New();
    Global<NumProcessPerNode>::Get()->set_value(1);
  }
  ~GlobalProcessCtxScope() {
    Global<NumProcessPerNode>::Delete();
    Global<ProcessCtx>::Delete();
  }
};

TEST(SequentialInstruction, front_seq_compute) {
  GlobalProcessCtxScope scope;
  auto vm_desc = ObjectMsgPtr<VmDesc>::New(TestUtil::NewVmResourceDesc().Get());
  TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(),
                                      {"NewObject", "ComputeRankFrontSeqCallback"});
  CachedObjectMsgAllocator allocator(20, 100);
  auto vm = ObjectMsgPtr<VirtualMachine>::NewFrom(&allocator, vm_desc.Get());
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
    *instruction->mutable_phy_instr_operand() =
        std::make_shared<vm::NoArgCbPhyInstrOperand>(Callback);
    list.EmplaceBack(std::move(instruction));
  }
  bool infer_finished = false;
  {
    auto instruction = NewInstruction("CtrlInferRankFrontSeqCallback");
    instruction->add_int64_operand(GlobalProcessCtx::Rank());
    const auto Callback = [&]() { infer_finished = true; };
    *instruction->mutable_phy_instr_operand() =
        std::make_shared<vm::NoArgCbPhyInstrOperand>(Callback);
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
    *instruction->mutable_phy_instr_operand() =
        std::make_shared<vm::NoArgCbPhyInstrOperand>(Callback);
    list.EmplaceBack(std::move(instruction));
  }
  BlockingCounter bc(1);
  std::thread t([&]() {
    while (!(infer_finished && compute_finished)) {
      vm->Schedule();
      OBJECT_MSG_LIST_FOR_EACH_PTR(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    }
    bc.Decrease();
  });
  vm->Receive(&list);
  bc.WaitUntilCntEqualZero();
  ASSERT_TRUE(is_666);
  ASSERT_TRUE(vm->Empty());
  t.join();
}

}  // namespace

}  // namespace test

}  // namespace vm
}  // namespace oneflow
