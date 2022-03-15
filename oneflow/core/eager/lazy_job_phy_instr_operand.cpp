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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"

namespace oneflow {
namespace vm {

namespace {

#ifdef WITH_CUDA
Maybe<LocalDepObject*> RawGetEagerNcclLocalDepObject(StreamRole stream_role) {
  // NOTE(chengcheng):
  //   Lazy Job instruction need mutual exclusion nccl with Eager nccl. However, when the number of
  //   processes is more than the number of physical GPUs, the following processes will make an
  //   error when using local rank to create a EagerNcclLocalDepObject, but we only need an legal
  //   device so we use device 0.
  const auto& device = JUST(Device::New("cpu", 0));
  const auto& stream = Stream::New(device, stream_role);
  const auto& local_dep_object = stream->mut_transport_local_dep_object();
  CHECK_OR_RETURN(local_dep_object.has_value());
  return JUST(local_dep_object);
}

static constexpr auto* GetEagerNcclLocalDepObject =
    DECORATE(&RawGetEagerNcclLocalDepObject, ThreadLocalCopiable);
#endif  // WITH_CUDA

}  // namespace

void LaunchLazyJobPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
  for (const auto& eager_blob_object : *param_blob_objects_) {
    DoEach(CHECK_JUST(eager_blob_object->compute_local_dep_object()));
  }
  DoEach(GetStaticGlobalTransportLocalDepObject());
}

}  // namespace vm
}  // namespace oneflow
