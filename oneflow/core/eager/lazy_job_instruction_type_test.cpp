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
#define private public
#include <thread>
#include <chrono>
#include <atomic>
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/buffer.h"
#include "oneflow/core/job/job_instance.h"

namespace oneflow {
namespace vm {
namespace test {

using InstructionMsgList = OBJECT_MSG_LIST(vm::InstructionMsg, instr_msg_link);

class NoArgNoRetMockNNGraph : public NNGraphIf {
 public:
  NoArgNoRetMockNNGraph(const std::string& job_name) : job_name_(job_name) {}
  ~NoArgNoRetMockNNGraph() override = default;

  const std::string& job_name() const override { return job_name_; }
  const std::vector<std::string>& inputs_op_names() const override {
    static std::vector<std::string> empty;
    return empty;
  }
  const std::vector<std::string>& outputs_op_names() const override {
    static std::vector<std::string> empty;
    return empty;
  }

 private:
  const std::string job_name_;
};

TEST(RunLazyJobInstructionType, simple) {
  vm::TestResourceDescScope resource_scope(0, 1);
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"RunLazyJob"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  Global<BufferMgr<std::shared_ptr<JobInstance>>>::New();
  const std::string job_name("test_job");
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
  buffer_mgr->NewBuffer(GetSourceTickBufferName(job_name), 128);
  std::thread enter_thread([&]() {
    std::shared_ptr<JobInstance> test_job_instance;
    auto* buffer = buffer_mgr->Get(GetSourceTickBufferName(job_name));
    while (buffer->Receive(&test_job_instance) == kBufferStatusSuccess) {
      // Do nothing
    }
  });
  buffer_mgr->NewBuffer(GetCallbackNotifierBufferName(job_name), 128);
  std::thread leave_thread([&]() {
    std::shared_ptr<JobInstance> test_job_instance;
    auto* buffer = buffer_mgr->Get(GetCallbackNotifierBufferName(job_name));
    while (buffer->Receive(&test_job_instance) == kBufferStatusSuccess) {
      test_job_instance->Finish();
    }
  });
  InstructionMsgList list;
  vm::cfg::EagerSymbolList eager_symbol_list;
  InstructionsBuilder instructions_builder(nullptr, &list, &eager_symbol_list);
  {
    static const auto& empty_list =
        std::make_shared<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>();
    const auto& nn_graph = std::make_shared<NoArgNoRetMockNNGraph>(job_name);
    CHECK_JUST(instructions_builder.RunLazyJob(empty_list, empty_list, empty_list, nn_graph));
    CHECK_JUST(instructions_builder.RunLazyJob(empty_list, empty_list, empty_list, nn_graph));
  }
  ASSERT_EQ(list.size(), 2);
  CHECK_JUST(vm->Receive(&list));
  auto* vm_ptr = vm.Mutable();
  std::thread scheduler_thread([vm_ptr]() {
    while (!vm_ptr->Empty()) {
      vm_ptr->Schedule();
      OBJECT_MSG_LIST_FOR_EACH_PTR(vm_ptr->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    }
  });
  scheduler_thread.join();
  buffer_mgr->Get(GetSourceTickBufferName(job_name))->Close();
  buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Close();
  leave_thread.join();
  enter_thread.join();
  Global<BufferMgr<std::shared_ptr<JobInstance>>>::Delete();
}

TEST(RunLazyJobInstructionType, wait_for_another_job_finished) {
  vm::TestResourceDescScope resource_scope(0, 1);
  auto vm_desc = ObjectMsgPtr<vm::VmDesc>::New(vm::TestUtil::NewVmResourceDesc().Get());
  vm::TestUtil::AddStreamDescByInstrNames(vm_desc.Mutable(), {"RunLazyJob"});
  auto vm = ObjectMsgPtr<vm::VirtualMachine>::New(vm_desc.Get());
  Global<BufferMgr<std::shared_ptr<JobInstance>>>::New();
  const std::string job_name0("test_job0");
  const std::string job_name1("test_job1");
  std::atomic<bool> flag_enter_thread0(false);
  std::atomic<int> count_enter_thread0(0);
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
  buffer_mgr->NewBuffer(GetSourceTickBufferName(job_name0), 128);
  buffer_mgr->NewBuffer(GetSourceTickBufferName(job_name1), 128);
  std::thread enter_thread0([&]() {
    while (!flag_enter_thread0) {}
    std::shared_ptr<JobInstance> test_job_instance;
    auto* buffer = buffer_mgr->Get(GetSourceTickBufferName(job_name0));
    while (buffer->Receive(&test_job_instance) == kBufferStatusSuccess) { ++count_enter_thread0; }
  });
  std::atomic<bool> flag_enter_thread1(false);
  std::atomic<int> count_enter_thread1(0);
  std::thread enter_thread1([&]() {
    while (!flag_enter_thread1) {}
    std::shared_ptr<JobInstance> test_job_instance;
    auto* buffer = buffer_mgr->Get(GetSourceTickBufferName(job_name1));
    while (buffer->Receive(&test_job_instance) == kBufferStatusSuccess) { ++count_enter_thread1; }
  });
  buffer_mgr->NewBuffer(GetCallbackNotifierBufferName(job_name0), 128);
  buffer_mgr->NewBuffer(GetCallbackNotifierBufferName(job_name1), 128);
  std::atomic<bool> flag_leave_thread0(false);
  std::atomic<int> count_leave_thread0(0);
  std::thread leave_thread0([&]() {
    while (!flag_leave_thread0) {}
    std::shared_ptr<JobInstance> test_job_instance;
    auto* buffer = buffer_mgr->Get(GetCallbackNotifierBufferName(job_name0));
    while (buffer->Receive(&test_job_instance) == kBufferStatusSuccess) {
      ++count_leave_thread0;
      test_job_instance->Finish();
    }
  });
  std::atomic<bool> flag_leave_thread1(false);
  std::atomic<int> count_leave_thread1(0);
  std::thread leave_thread1([&]() {
    while (!flag_leave_thread1) {}
    std::shared_ptr<JobInstance> test_job_instance;
    auto* buffer = buffer_mgr->Get(GetCallbackNotifierBufferName(job_name1));
    while (buffer->Receive(&test_job_instance) == kBufferStatusSuccess) {
      ++count_leave_thread1;
      test_job_instance->Finish();
    }
  });
  buffer_mgr->NewBuffer(GetForeignInputBufferName(job_name0), 128);
  buffer_mgr->NewBuffer(GetForeignInputBufferName(job_name1), 128);
  buffer_mgr->NewBuffer(GetForeignOutputBufferName(job_name0), 128);
  buffer_mgr->NewBuffer(GetForeignOutputBufferName(job_name1), 128);
  InstructionMsgList list;
  vm::cfg::EagerSymbolList eager_symbol_list;
  InstructionsBuilder instructions_builder(nullptr, &list, &eager_symbol_list);
  int num_job0_instance = 2;
  int num_job1_instance = 3;
  {
    static const auto& empty_list =
        std::make_shared<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>();
    const auto& nn_graph0 = std::make_shared<NoArgNoRetMockNNGraph>(job_name0);
    const auto& nn_graph1 = std::make_shared<NoArgNoRetMockNNGraph>(job_name1);
    for (int i = 0; i < num_job0_instance; ++i) {
      CHECK_JUST(instructions_builder.RunLazyJob(empty_list, empty_list, empty_list, nn_graph0));
    }
    for (int i = 0; i < num_job1_instance; ++i) {
      CHECK_JUST(instructions_builder.RunLazyJob(empty_list, empty_list, empty_list, nn_graph1));
    }
  }
  ASSERT_EQ(list.size(), num_job0_instance + num_job1_instance);
  CHECK_JUST(vm->Receive(&list));
  ASSERT_EQ(vm->pending_msg_list().size(), num_job0_instance + num_job1_instance);
  auto* vm_ptr = vm.Mutable();
  std::thread scheduler_thread([vm_ptr]() {
    while (!vm_ptr->Empty()) {
      vm_ptr->Schedule();
      OBJECT_MSG_LIST_FOR_EACH_PTR(vm_ptr->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    }
  });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_EQ(count_enter_thread0, 0);
  ASSERT_EQ(count_leave_thread0, 0);
  ASSERT_EQ(count_enter_thread1, 0);
  ASSERT_EQ(count_leave_thread1, 0);
  flag_enter_thread0 = true;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_EQ(count_enter_thread0, num_job0_instance);
  ASSERT_EQ(count_leave_thread0, 0);
  ASSERT_EQ(count_enter_thread1, 0);
  ASSERT_EQ(count_leave_thread1, 0);
  flag_enter_thread1 = true;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_EQ(count_enter_thread0, num_job0_instance);
  ASSERT_EQ(count_leave_thread0, 0);
  ASSERT_EQ(count_enter_thread1, 0);
  ASSERT_EQ(count_leave_thread1, 0);
  flag_leave_thread0 = true;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_EQ(count_enter_thread0, num_job0_instance);
  ASSERT_EQ(count_leave_thread0, num_job0_instance);
  ASSERT_EQ(count_enter_thread1, num_job1_instance);
  ASSERT_EQ(count_leave_thread1, 0);
  flag_leave_thread1 = true;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_EQ(count_enter_thread0, num_job0_instance);
  ASSERT_EQ(count_leave_thread0, num_job0_instance);
  ASSERT_EQ(count_enter_thread1, num_job1_instance);
  ASSERT_EQ(count_leave_thread1, num_job1_instance);
  scheduler_thread.join();
  buffer_mgr->Get(GetSourceTickBufferName(job_name0))->Close();
  buffer_mgr->Get(GetSourceTickBufferName(job_name1))->Close();
  buffer_mgr->Get(GetCallbackNotifierBufferName(job_name0))->Close();
  buffer_mgr->Get(GetCallbackNotifierBufferName(job_name1))->Close();
  leave_thread0.join();
  leave_thread1.join();
  enter_thread0.join();
  enter_thread1.join();
  Global<BufferMgr<std::shared_ptr<JobInstance>>>::Delete();
}

}  // namespace test
}  // namespace vm
}  // namespace oneflow
