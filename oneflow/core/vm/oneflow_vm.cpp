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
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

OneflowVM::OneflowVM(const Resource& resource, int64_t this_machine_id)
    : vm_(ObjectMsgPtr<vm::VirtualMachine>::New(vm::MakeVmDesc(resource, this_machine_id).Get())) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_->mut_thread_ctx_list(), thread_ctx) {
    auto thread_pool = std::make_unique<ThreadPool>(1);
    CHECK(thread_ctx2thread_pool_.emplace(thread_ctx, std::move(thread_pool)).second);
  }
  loop_thread_ = std::thread(&OneflowVM::Loop, this);
  exiting_ = false;
}

OneflowVM::~OneflowVM() {
  exiting_ = true;
  loop_thread_.join();
}

void OneflowVM::Loop() {
  auto* vm = mut_vm();
  while (!exiting_) {
    if (Global<ResourceDesc, ForSession>::Get()
        && Global<ResourceDesc, ForSession>::Get()->async_eager_execution()) {
      std::lock_guard<std::mutex> lock(mutex_);
      vm->Schedule();
      TryReceiveAndRun();
    }
  }
}

void OneflowVM::Sync() {
  auto* vm = mut_vm();
  while (true) {
    // object msg is not thread safe so add a mutex here
    std::lock_guard<std::mutex> lock(mutex_);
    if (vm->Empty()) { break; }
  }
}

void OneflowVM::TryReceiveAndRun() {
  for (auto& pair : thread_ctx2thread_pool_) {
    vm::ThreadCtx* thread_ctx = pair.first;
    if (thread_ctx->mut_pending_instruction_list()->Empty()) { continue; }
    pair.second->AddWork([thread_ctx]() { thread_ctx->TryReceiveAndRun(); });
  }
}

}  // namespace oneflow
