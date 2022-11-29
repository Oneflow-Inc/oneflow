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
#ifndef ONEFLOW_CORE_ODB_ODB_H_
#define ONEFLOW_CORE_ODB_ODB_H_

namespace oneflow {
namespace odb {

enum ThreadType {
  kInvalidThreadType = 0,
  kNormalThreadType = 1,
  kSchedulerThreadType = 2,
  kWorkerThreadType = 3,
};

void SetThreadTypeBreakpoint(ThreadType thread_type);
void ClearThreadTypeBreakpoint(ThreadType thread_type);

void InitThisThreadType(ThreadType thread_type);

class BreakpointRange final {
 public:
  BreakpointRange();
  ~BreakpointRange();
};

enum BreakpointRangeMode {
  kEnableBreakpointRange = 0,
  kDisableBreakpointRange = 1,
};

class BreakpointRangeModeGuard final {
 public:
  explicit BreakpointRangeModeGuard(BreakpointRangeMode mode);
  ~BreakpointRangeModeGuard();

  static BreakpointRangeMode Current();

 private:
  BreakpointRangeMode prev_mode_;
};

void SetBreakpointThreadType(ThreadType thread_type);

template<ThreadType thread_type>
bool GetStopVMFlag();
template<ThreadType thread_type>
void StopVMThread();
template<ThreadType thread_type>
void RestartVMThread();

template<ThreadType thread_type>
void TryBlockVMThread();

}  // namespace odb
}  // namespace oneflow

#endif  // ONEFLOW_CORE_ODB_ODB_H_
