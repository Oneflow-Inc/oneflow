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
#include "oneflow/core/framework/framework.h"
#ifdef OF_PLATFORM_POSIX
#include <unistd.h>
#include <sys/wait.h>
#endif  // OF_PLATFORM_POSIX
#include <thread>

namespace oneflow {

namespace {

class TestForkOnWorkerThreadKernel final : public user_op::OpKernel {
 public:
  TestForkOnWorkerThreadKernel() = default;
  ~TestForkOnWorkerThreadKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
#ifdef OF_PLATFORM_POSIX
    const auto& TestFork = [] {
      pid_t pid = fork();
      if (pid == 0) {
        // We're the child process.
        _exit(0);
      } else if (pid > 0) {
        // We're the parent process
        int status;
        waitpid(pid, &status, 0);
      } else {
        // Do nothing.
      }
    };
    // test fork on current thread.
    TestFork();
    // test fork on another thread.
    std::thread(TestFork).join();
#endif  // OF_PLATFORM_POSIX
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("test_fork_on_worker_thread")
    .SetCreateFn<TestForkOnWorkerThreadKernel>()
    .SetIsMatchedHob(user_op::HobTrue());
}  // namespace

}  // namespace oneflow
