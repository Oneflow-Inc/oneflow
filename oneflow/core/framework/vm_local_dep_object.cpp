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
#include "oneflow/core/framework/vm_local_dep_object.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/vm_object.msg.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {
namespace vm {

void LocalDepObject::__Init__(const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  vm::ObjectId object_id = vm::IdUtil::NewPhysicalValueObjectId(GlobalProcessCtx::Rank());
  int64_t global_device_id = 0;
  {
    CHECK_EQ(parallel_desc->parallel_num(), 1);
    int64_t machine_id = CHECK_JUST(parallel_desc->MachineId4ParallelId(0));
    CHECK_EQ(machine_id, GlobalProcessCtx::Rank());
    int64_t device_id = CHECK_JUST(parallel_desc->DeviceId4ParallelId(0));
    const auto& vm = Global<OneflowVM>::Get()->vm();
    CHECK_EQ(vm.this_machine_id(), machine_id);
    global_device_id = vm.this_start_global_device_id() + device_id;
  }
  mutable_logical_object()->__Init__(object_id,
                                     std::const_pointer_cast<ParallelDesc>(parallel_desc));
  mutable_mirrored_object()->__Init__(mutable_logical_object(), global_device_id);
}

namespace {

using LocalDepObjectFreeList = OBJECT_MSG_LIST(LocalDepObject, free_link);
using LocalDepObjectZombieList = OBJECT_MSG_LIST(LocalDepObject, zombie_link);
using LocalDepObjectMutexedZombieList = OBJECT_MSG_MUTEXED_LIST(LocalDepObject, zombie_link);

LocalDepObjectFreeList* ThreadLocalMutFreeList4ParallelDesc(const ParallelDesc& parallel_desc) {
  thread_local static HashMap<ParallelDesc, LocalDepObjectFreeList> pd2free_list;
  return &pd2free_list[parallel_desc];
}

LocalDepObjectMutexedZombieList* StaticMutLocalDepObjectMutexedZombieList() {
  static LocalDepObjectMutexedZombieList zombie_list;
  return &zombie_list;
}

void TryMoveFromZombieListToFreeList() {
  thread_local static LocalDepObjectZombieList zombie_list;
  if (zombie_list.empty()) { StaticMutLocalDepObjectMutexedZombieList()->MoveTo(&zombie_list); }
  static const size_t kTryCnt = 8;
  size_t try_cnt = kTryCnt;
  OBJECT_MSG_LIST_FOR_EACH(&zombie_list, zombie_object) {
    zombie_list.Erase(zombie_object.Mutable());
    size_t ref_cnt = zombie_object->ref_cnt();
    if (ref_cnt == 1 /* hold by `zombie_object` only */) {
      CHECK_EQ(zombie_object->mirrored_object().rw_mutexed_object().ref_cnt(), 1);
      CHECK(zombie_object->mirrored_object().rw_mutexed_object().access_list().empty());
      const auto& parallel_desc = *zombie_object->logical_object().parallel_desc();
      auto* thread_local_free_list = ThreadLocalMutFreeList4ParallelDesc(parallel_desc);
      thread_local_free_list->EmplaceBack(std::move(zombie_object));
    } else {
      CHECK_GT(ref_cnt, 1);
      zombie_list.EmplaceBack(std::move(zombie_object));
    }
    if (--try_cnt < 0) { break; }
  }
}

ObjectMsgPtr<LocalDepObject> GetRecycledLocalDepObject(const ParallelDesc& parallel_desc) {
  auto* thread_local_free_list = ThreadLocalMutFreeList4ParallelDesc(parallel_desc);
  if (thread_local_free_list->empty()) {
    TryMoveFromZombieListToFreeList();
    if (thread_local_free_list->empty()) { return nullptr; }
  }
  ObjectMsgPtr<LocalDepObject> object = thread_local_free_list->Begin();
  thread_local_free_list->Erase(object.Mutable());
  CHECK_EQ(object->ref_cnt(), 1);  // hold by `object` only
  return std::move(object);
}

void MoveLocalDepObjectToZombieList(ObjectMsgPtr<LocalDepObject>&& local_dep_object) {
  static const size_t kGroupSize = 16;
  thread_local static LocalDepObjectZombieList zombie_list;
  zombie_list.EmplaceBack(std::move(local_dep_object));
  if (zombie_list.size() >= kGroupSize) {
    StaticMutLocalDepObjectMutexedZombieList()->MoveFrom(&zombie_list);
  }
}

}  // namespace
}  // namespace vm

VmLocalDepObject::VmLocalDepObject(const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  local_dep_object_ = vm::GetRecycledLocalDepObject(*parallel_desc);
  if (!local_dep_object_) {
    local_dep_object_ = ObjectMsgPtr<vm::LocalDepObject>::New(parallel_desc);
  }
}

VmLocalDepObject::~VmLocalDepObject() {
  vm::MoveLocalDepObjectToZombieList(std::move(local_dep_object_));
}

}  // namespace oneflow
