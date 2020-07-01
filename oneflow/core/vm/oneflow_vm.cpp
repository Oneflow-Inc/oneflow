#include "oneflow/core/vm/oneflow_vm.h"

namespace oneflow {

OneflowVM::OneflowVM(const Resource& resource, int64_t this_machine_id)
    : vm_(ObjectMsgPtr<vm::VirtualMachine>::New(vm::MakeVmDesc(resource, this_machine_id).Get())) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_->mut_thread_ctx_list(), thread_ctx) {
    auto thread_pool = std::make_unique<ThreadPool>(1);
    CHECK(thread_ctx2thread_pool_.emplace(thread_ctx, std::move(thread_pool)).second);
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
