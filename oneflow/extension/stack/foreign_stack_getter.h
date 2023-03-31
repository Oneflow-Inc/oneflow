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
#ifndef ONEFLOW_EXTENSION_STACK_STACK_GETTER_H_
#define ONEFLOW_EXTENSION_STACK_STACK_GETTER_H_

#include <cstdint>
#include <utility>
#include "oneflow/core/common/thread_local_guard.h"

namespace oneflow {

class Frame {
 public:
  virtual ~Frame() = default;
};

using ForeignFrameThreadLocalGuard = ThreadLocalGuard<std::shared_ptr<Frame>>;

class ForeignStackGetter {
 public:
  virtual ~ForeignStackGetter() = default;
  virtual std::shared_ptr<Frame> GetCurrentFrame() const = 0;
  virtual std::string GetFormattedStack(std::shared_ptr<Frame> frame) const = 0;
};
}  // namespace oneflow

#endif  // ONEFLOW_EXTENSION_STACK_STACK_GETTER_H_
