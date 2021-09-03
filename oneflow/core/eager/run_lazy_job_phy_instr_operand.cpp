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
#include "oneflow/core/eager/run_lazy_job_phy_instr_operand.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {
namespace vm {

void RunLazyJobPhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  for (const auto& input : *inputs()) {
    DoEach(nullptr, CHECK_JUST(input->compute_local_dep_object())->mut_mirrored_object());
  }
}

namespace {

Maybe<LocalDepObject*> RawGetEagerNcclLocalDepObject(const std::string& type) {
  const auto& device = JUST(Device::New(type));
  const auto& local_dep_object = device->mut_transport_local_dep_object();
  CHECK_OR_RETURN(local_dep_object.has_value());
  return JUST(local_dep_object.value());
}

}  // namespace

static constexpr auto* GetEagerNcclLocalDepObject =
    DECORATE(&RawGetEagerNcclLocalDepObject, ThreadLocalCopiable);

void RunLazyJobPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
#ifdef WITH_CUDA
  auto* sync_launched_nccl = CHECK_JUST(GetEagerNcclLocalDepObject("sync_launched_nccl"));
  auto* async_launched_nccl = CHECK_JUST(GetEagerNcclLocalDepObject("async_launched_nccl"));
  CHECK_EQ(sync_launched_nccl, async_launched_nccl);
  DoEach(nullptr, async_launched_nccl->mut_mirrored_object());
#endif  // WITH_CUDA
  for (const auto& parameter : *parameters()) {
    DoEach(nullptr, CHECK_JUST(parameter->compute_local_dep_object())->mut_mirrored_object());
  }
}

void RunLazyJobPhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  // TODO(lixinqi): move partial of outputs into ForEachMutMirroredObject if shape infered before
  // compute.
  for (const auto& output : *outputs()) {
    DoEach(nullptr, CHECK_JUST(output->compute_local_dep_object())->mut_mirrored_object());
  }
}

}  // namespace vm
}  // namespace oneflow
