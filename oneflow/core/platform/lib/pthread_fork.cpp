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
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/sync_vm_mode_guard.h"

namespace oneflow {

namespace pthread_fork {

static bool is_fork = false;

bool IsForkedSubProcess() { return is_fork; }
static void SetIsForkedSubProcess() { is_fork = true; }

namespace {
void CurrentRankVmSync() {
  if (SyncVmModeGuard::IsCurrentSyncVmMode()) { return; }
  // Instructions in forked subprocesses are not dispatched to vm,
  // so no need to sync vm in these processes.
  if (!is_fork && Singleton<VirtualMachine>::Get() != nullptr) {
    CHECK_JUST(vm::CurrentRankSync());
  }
}
}  // namespace

void RegisterForkCallback() { pthread_atfork(&CurrentRankVmSync, nullptr, &SetIsForkedSubProcess); }
COMMAND(RegisterForkCallback());

const char* kOfCudaNotSupportInForkedSubProcess =
    "Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you "
    "must add 'multiprocessing.set_start_method(\"spawn\")' in '__main__' if you are using "
    "Python's multiprocessing";

}  // namespace pthread_fork

}  // namespace oneflow
