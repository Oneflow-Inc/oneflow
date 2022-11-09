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
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/kernel/kernel_observer.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

class Blob;
class JobDesc;

class KernelState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelState);
  KernelState() = default;
  virtual ~KernelState() = default;
};

class KernelContext : public KernelObserver {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelContext);
  KernelContext() = default;
  virtual ~KernelContext() = default;

  virtual ep::Stream* stream() const = 0;
  virtual Blob* BnInOp2Blob(const std::string& bn) const = 0;
  virtual const std::shared_ptr<KernelState>& state() const = 0;
  virtual void set_state(std::shared_ptr<KernelState> state) = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_
