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

using PoolLocalDepObjectList = OBJECT_MSG_LIST(LocalDepObject, pool_link);
using StoredLocalDepObjectList = OBJECT_MSG_LIST(LocalDepObject, occupied_link);
using OccupiedLocalDepObjectList = OBJECT_MSG_LIST(LocalDepObject, occupied_link);

PoolLocalDepObjectList* RawThreadLocalPoolLocalDepObjectList(Symbol<Device> device) {
  static thread_local PoolLocalDepObjectList pool_list;
  return &pool_list;
}
static constexpr auto* ThreadLocalPoolLocalDepObjectList = DECORATE(&RawThreadLocalPoolLocalDepObjectList, ThreadLocal);

StoredLocalDepObjectList* RawThreadLocalStoredLocalDepObjectList(Symbol<Device> device) {
  static thread_local StoredLocalDepObjectList stored_list;
  return &stored_list;
}
static constexpr auto* ThreadLocalStoredLocalDepObjectList = DECORATE(&RawThreadLocalStoredLocalDepObjectList, ThreadLocal);

OccupiedLocalDepObjectList* RawThreadLocalOccupiedLocalDepObjectList(Symbol<Device> device) {
  static thread_local OccupiedLocalDepObjectList occupied_list;
  return &occupied_list;
}
static constexpr auto* ThreadLocalOccupiedLocalDepObjectList = DECORATE(&RawThreadLocalOccupiedLocalDepObjectList, ThreadLocal);

}  // namespace

Maybe<LocalDepObject*> GetLocalDepObjectFromDevicePool(Symbol<Device> device) {
  ObjectMsgPtr<LocalDepObject> local_dep_object;
  auto* pool_list = ThreadLocalPoolLocalDepObjectList(device);
  auto* stored_list = ThreadLocalStoredLocalDepObjectList(device);
  if (!pool_list->empty()) {
    // When running stable, fetch recycled local_dep_object from pool_list which acting as a
    // object pool.
    local_dep_object = pool_list->PopFront();
  } else if (!stored_list->empty()) {
    // When running unstable, try fetch local_dep_object from stored_list
    local_dep_object = stored_list->PopFront();
  } else {
    // When running unstable and no stored objects, directly new LocalDepObject
    local_dep_object = *JUST(LocalDepObject::New(*device));
  }
  auto* ptr = local_dep_object.Mutable();
  CHECK_OR_RETURN(local_dep_object->is_pool_link_empty());
  CHECK_OR_RETURN(local_dep_object->is_occupied_link_empty());
  ThreadLocalOccupiedLocalDepObjectList(device)->EmplaceBack(std::move(local_dep_object));
  return ptr;
}

Maybe<void> PutLocalDepObjectToDevicePool(Symbol<Device> device, LocalDepObject* local_dep_object) {
  CHECK_OR_RETURN(local_dep_object->is_pool_link_empty());
  CHECK_OR_RETURN(local_dep_object->is_stored_link_empty());
  CHECK_OR_RETURN(!local_dep_object->is_occupied_link_empty());
  auto* pool_list = ThreadLocalPoolLocalDepObjectList(device);
  const auto& pool_size = JUST(device->instr_local_dep_object_pool_size());
  // Keep pool_list->size() not bigger than pool_size
  if (pool_list->size() < pool_size) {
    pool_list->PushBack(local_dep_object);
  } else {
    ThreadLocalStoredLocalDepObjectList(device)->PushBack(local_dep_object);
  }
  ThreadLocalOccupiedLocalDepObjectList(device)->Erase(local_dep_object);
  return Maybe<void>::Ok();
}

Maybe<LocalDepObject*> GetLocalDepObject4Device(const Device& device) {
  static constexpr auto* GetObj = DECORATE(&LocalDepObject::New, StaticGlobalCopiable);
  return JUST(GetObj(device))->Mutable();
}
}  // namespace oneflow
