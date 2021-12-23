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
#include <sstream>
#include <vector>
#include <set>
#include <glog/logging.h>
#include "oneflow/core/common/smart_monkey.h"

namespace oneflow {
namespace chaos {

std::vector<const std::string*>* MutThreadLocalSourceCodePositionStack() {
  thread_local std::vector<const std::string*> stack;
  return &stack;
}

const std::vector<const std::string*>& ThreadLocalSourceCodePositionStack() {
  return *MutThreadLocalSourceCodePositionStack();
}

SourceCodePositionScope::SourceCodePositionScope(const std::string* src_code_pos) {
  if (ThreadLocalEnableChaos()) {
    MutThreadLocalSourceCodePositionStack()->push_back(src_code_pos);
  }
}

SourceCodePositionScope::~SourceCodePositionScope() {
  if (ThreadLocalEnableChaos()) { MutThreadLocalSourceCodePositionStack()->pop_back(); }
}

bool SmartMonkey::Fail() {
  if (!ThreadLocalEnableChaos()) { return false; }
  return stacks_.insert(ThreadLocalSourceCodePositionStack()).second;
}

}  // namespace chaos
}  // namespace oneflow
