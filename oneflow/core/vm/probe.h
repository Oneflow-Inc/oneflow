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

#ifndef ONEFLOW_CORE_VM_PROBE_H_
#define ONEFLOW_CORE_VM_PROBE_H_

#include "oneflow/core/intrusive/intrusive.h"

namespace oneflow {
namespace vm {

class VirtualMachineEngine;

class Probe final : public intrusive::Base {
 public:
  Probe(const Probe&) = delete;
  Probe(Probe&&) = delete;

  Probe() = default;
  ~Probe() = default;

  // return true when this Probe should be removed from virtual machine.
  using ProbeFunction = std::function<bool(VirtualMachineEngine*)>;

  void __Init__(const ProbeFunction& probe_function) { probe_function_ = probe_function; }

  bool probe(VirtualMachineEngine* vm) const {
    CHECK(static_cast<bool>(probe_function_));
    return probe_function_(vm);
  }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  // fields
  intrusive::Ref intrusive_ref_;
  ProbeFunction probe_function_;

 public:
  // hooks
  intrusive::ListHook probe_hook_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_PROBE_H_
