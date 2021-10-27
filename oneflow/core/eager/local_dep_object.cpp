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
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/static_global.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/vm/virtual_machine.h"
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
  mut_logical_object()->__Init__(object_id, std::const_pointer_cast<ParallelDesc>(parallel_desc));
  mut_mirrored_object()->__Init__(mut_logical_object(), global_device_id);
  return Maybe<void>::Ok();
}

Maybe<intrusive::shared_ptr<LocalDepObject>> LocalDepObject::New(const Device& device) {
  auto local_dep_obj = intrusive::make_shared<LocalDepObject>();
  JUST(local_dep_obj.Mutable()->Init(device));
  return local_dep_obj;
}

namespace {

using PoolLocalDepObjectList = intrusive::List<INTRUSIVE_FIELD(LocalDepObject, pool_hook_)>;

std::shared_ptr<PoolLocalDepObjectList> RawThreadLocalPoolLocalDepObjectList(
    const std::string& of_device_type) {
  return std::make_shared<PoolLocalDepObjectList>();
}

static constexpr auto* ThreadLocalPoolLocalDepObjectList =
    DECORATE(&RawThreadLocalPoolLocalDepObjectList, ThreadLocalCopiable);

using LifetimeLocalDepObjectList =
    intrusive::MutexedList<INTRUSIVE_FIELD(LocalDepObject, lifetime_hook_)>;
std::shared_ptr<LifetimeLocalDepObjectList> RawGlobalLifetimeLocalDepObjectList(
    const std::string& of_device_type) {
  return std::make_shared<LifetimeLocalDepObjectList>();
}

static constexpr auto* GlobalLifetimeLocalDepObjectList =
    DECORATE(&RawGlobalLifetimeLocalDepObjectList, StaticGlobalCopiable);

std::shared_ptr<intrusive::List<INTRUSIVE_FIELD(LocalDepObject, pool_hook_)>>
RawThreadLocalOverflowedLocalDepObjectList(const std::string& of_device_type) {
  return std::make_shared<intrusive::List<INTRUSIVE_FIELD(LocalDepObject, pool_hook_)>>();
}

static constexpr auto* ThreadLocalOverflowedLocalDepObjectList =
    DECORATE(&RawThreadLocalOverflowedLocalDepObjectList, StaticGlobalCopiable);

std::shared_ptr<intrusive::MutexedList<INTRUSIVE_FIELD(LocalDepObject, pool_hook_)>>
RawGlobalOverflowedLocalDepObjectList(const std::string& of_device_type) {
  return std::make_shared<intrusive::MutexedList<INTRUSIVE_FIELD(LocalDepObject, pool_hook_)>>();
}

static constexpr auto* GlobalOverflowedLocalDepObjectList =
    DECORATE(&RawGlobalOverflowedLocalDepObjectList, StaticGlobalCopiable);

intrusive::shared_ptr<LocalDepObject> GetOverflowedLocalDepObject(Symbol<Device> device) {
  const std::string& of_dev_type = CHECK_JUST(device->of_type());
  const auto& thread_local_overflowed_list = ThreadLocalOverflowedLocalDepObjectList(of_dev_type);
  if (thread_local_overflowed_list->empty()) {
    GlobalOverflowedLocalDepObjectList(of_dev_type)->MoveTo(thread_local_overflowed_list.get());
  }
  return thread_local_overflowed_list->PopFront();
}

Maybe<void> PutOverflowedLocalDepObject(Symbol<Device> device, LocalDepObject* local_dep_object) {
  const std::string& of_dev_type = JUST(device->of_type());
  const auto& thread_local_overflowed_list = ThreadLocalOverflowedLocalDepObjectList(of_dev_type);
  static constexpr int kThreadLocalOverflowSize = 1024;
  if (thread_local_overflowed_list->size() > kThreadLocalOverflowSize) {
    GlobalOverflowedLocalDepObjectList(of_dev_type)->MoveFrom(thread_local_overflowed_list.get());
  }
  thread_local_overflowed_list->PushBack(local_dep_object);
  return Maybe<void>::Ok();
}

static constexpr size_t kLowWaterMark = 200;
static constexpr size_t kHighWaterMark = 300;

Maybe<void> TryFillPoolToLowWaterMark(Symbol<Device> device) {
  const std::string& of_dev_type = JUST(device->of_type());
  const auto& pool_list = ThreadLocalPoolLocalDepObjectList(of_dev_type);
  for (int i = pool_list->size(); i < kLowWaterMark; ++i) {
    intrusive::shared_ptr<LocalDepObject> local_dep_object = GetOverflowedLocalDepObject(device);
    if (!local_dep_object) {
      local_dep_object = *JUST(LocalDepObject::New(*device));
      GlobalLifetimeLocalDepObjectList(of_dev_type)->PushBack(local_dep_object.Mutable());
    }
    pool_list->PushBack(local_dep_object.Mutable());
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<LocalDepObject*> GetLocalDepObjectFromDevicePool(Symbol<Device> device) {
  const std::string& of_dev_type = JUST(device->of_type());
  JUST(TryFillPoolToLowWaterMark(device));
  const auto& pool_list = ThreadLocalPoolLocalDepObjectList(of_dev_type);
  CHECK_GE_OR_RETURN(pool_list->size(), kLowWaterMark);
  CHECK_LE_OR_RETURN(pool_list->size(), kHighWaterMark);
  intrusive::shared_ptr<LocalDepObject> local_dep_object = pool_list->PopFront();
  CHECK_NOTNULL_OR_RETURN(local_dep_object.Mutable());
  CHECK_OR_RETURN(local_dep_object->pool_hook().empty());
  CHECK_OR_RETURN(!local_dep_object->lifetime_hook().empty());
  return local_dep_object.Mutable();
}

Maybe<void> PutLocalDepObjectToDevicePool(Symbol<Device> device, LocalDepObject* local_dep_object) {
  const std::string& of_dev_type = JUST(device->of_type());
  CHECK_OR_RETURN(local_dep_object->pool_hook().empty());
  CHECK_OR_RETURN(!local_dep_object->lifetime_hook().empty());
  const auto& pool_list = ThreadLocalPoolLocalDepObjectList(of_dev_type);
  if (pool_list->size() < kHighWaterMark) {
    pool_list->PushBack(local_dep_object);
  } else {
    JUST(PutOverflowedLocalDepObject(device, local_dep_object));
  }
  CHECK_GE_OR_RETURN(pool_list->size(), kLowWaterMark);
  CHECK_LE_OR_RETURN(pool_list->size(), kHighWaterMark);
  return Maybe<void>::Ok();
}

Maybe<LocalDepObject*> GetLocalDepObject4Device(const Device& device) {
  static constexpr auto* GetObj = DECORATE(&LocalDepObject::New, StaticGlobalCopiable);
  return JUST(GetObj(device))->Mutable();
}
}  // namespace oneflow
