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
#include "oneflow/core/eager/lazy_job_phy_instr_operand.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {
namespace vm {

namespace {

Maybe<ObjectMsgPtr<LocalDepObject>> RawGetLocalDepObject(const std::string& type) {
  const auto& device = JUST(Device::New(type));
  return LocalDepObject::New(*device);
}

}  // namespace

static constexpr auto* GetLocalDepObject =
    DECORATE(&RawGetLocalDepObject, ThreadLocalCopiable);

void LaunchLazyJobPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  auto dep_object = *CHECK_JUST(GetLocalDepObject("cpu"));
  DoEach(nullptr, dep_object->mut_mirrored_object());
  // lifetime of parameters are managed by params_critical_section_ and nccl_critical_section_.
}

}  // namespace vm
}  // namespace oneflow
