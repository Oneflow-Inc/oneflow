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
#include <glog/logging.h>
#include "oneflow/core/profiler/odb.h"

namespace oneflow {
namespace odb {

namespace {

ThreadType* MutThreadLocalThreadType() {
  static thread_local ThreadType thread_type = kNormalThreadType;
  return &thread_type;
}

ThreadType GetThreadLocalThreadType() { return *MutThreadLocalThreadType(); }

}  // namespace

void InitThisThreadType(ThreadType thread_type) { *MutThreadLocalThreadType() = thread_type; }

namespace {

int64_t* MutThreadDepth() {
  static thread_local int64_t depth = 0;
  return &depth;
}

void DoNothing() {}

// anchor for gdb breakpoint
void BreakpointAnchor() { DoNothing(); }

}  // namespace

Guard::Guard() {
  depth_ = *MutThreadDepth();
  switch (GetThreadLocalThreadType()) {
    case kNormalThreadType: BreakpointAnchor(); break;
    case kSchedulerThreadType: BreakpointAnchor(); break;
    case kWorkerThreadType: BreakpointAnchor(); break;
  }
  ++*MutThreadDepth();
}

Guard::~Guard() {
  --*MutThreadDepth();
  switch (GetThreadLocalThreadType()) {
    case kNormalThreadType: BreakpointAnchor(); break;
    case kSchedulerThreadType: BreakpointAnchor(); break;
    case kWorkerThreadType: BreakpointAnchor(); break;
  }
  CHECK_EQ(depth_, *MutThreadDepth());
}

}  // namespace odb
}  // namespace oneflow
