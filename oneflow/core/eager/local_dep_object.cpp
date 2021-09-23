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
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/static_global.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/vm_object.msg.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {

Maybe<void> LocalDepObject::Init(const Device& device) {
  const auto& parallel_desc = JUST(Device::GetPlacement(device)).shared_from_symbol();
  vm::ObjectId object_id = vm::IdUtil::NewPhysicalValueObjectId(GlobalProcessCtx::Rank());
  int64_t global_device_id = 0;
  {
    CHECK_EQ(parallel_desc->parallel_num(), 1);
    int64_t machine_id = CHECK_JUST(parallel_desc->MachineId4ParallelId(0));
    CHECK_EQ(machine_id, GlobalProcessCtx::Rank());
    int64_t device_id = CHECK_JUST(parallel_desc->DeviceId4ParallelId(0));
    if (Global<OneflowVM>::Get() == nullptr) {
      global_device_id = 0;
    } else {
      const auto& vm = Global<OneflowVM>::Get()->vm();
      CHECK_EQ(vm.this_machine_id(), machine_id);
      global_device_id = vm.this_start_global_device_id() + device_id;
    }
  }
  mutable_logical_object()->__Init__(object_id,
                                     std::const_pointer_cast<ParallelDesc>(parallel_desc));
  mutable_mirrored_object()->__Init__(mutable_logical_object(), global_device_id);
  return Maybe<void>::Ok();
}

Maybe<ObjectMsgPtr<LocalDepObject>> LocalDepObject::New(const Device& device) {
  auto local_dep_obj = ObjectMsgPtr<LocalDepObject>::New();
  JUST(local_dep_obj.Mutable()->Init(device));
  return local_dep_obj;
}

namespace {

Maybe<std::vector<ObjectMsgPtr<LocalDepObject>>> RawGetStreamLocalDepObjectPool(Symbol<Stream> stream) {
  const auto pool = std::make_shared<std::vector<ObjectMsgPtr<LocalDepObject>>>();
  size_t pool_size = JUST(stream->stream_descriptor().local_dep_object_pool_size());
  pool->reserve(pool_size);
  for (int64_t i = 0; i < pool_size; ++i) {
    auto local_dep_object = *JUST(LocalDepObject::New(stream->device()));
    pool->push_back(local_dep_object);
  }
  return pool;
}

Maybe<std::vector<ObjectMsgPtr<LocalDepObject>>> RawGetDeviceLocalDepObjectPool(Symbol<Device> device) {
  const auto& stream = JUST(Stream::NewWithDefaultName(device));
  return JUST(RawGetStreamLocalDepObjectPool(stream));
}

static constexpr auto* GetStreamLocalDepObjectPool = DECORATE(&RawGetStreamLocalDepObjectPool, ThreadLocal);
static constexpr auto* GetDeviceLocalDepObjectPool = DECORATE(&RawGetDeviceLocalDepObjectPool, ThreadLocal);

template<typename T, typename GetPool>
Maybe<LocalDepObject*> GetLocalDepObjectFromPool<Stream>(Symbol<T> stream_or_device) {
  const auto& local_dep_object_pool = JUST(GetPool(stream_or_device));
  CHECK_OR_RETURN(!local_dep_object_pool->empty());
  size_t pool_size = local_dep_object_pool->size();
  static thread_local int64_t index = 0;
  return local_dep_object_pool->at(index++ % pool_size).Mutable();
}

}  // namespace

Maybe<LocalDepObject*> GetLocalDepObjectFromPool(Symbol<Stream> stream) {
  return GetLocalDepObjectFromPool<Stream, GetStreamLocalDepObjectPool>(stream);
}

Maybe<LocalDepObject*> GetLocalDepObjectFromPool(Symbol<Device> device) {
  return GetLocalDepObjectFromPool<Device, GetDeviceLocalDepObjectPool>(device);
}

}  // namespace oneflow
