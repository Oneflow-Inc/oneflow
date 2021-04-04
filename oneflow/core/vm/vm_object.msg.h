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
#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/object_msg/flat_msg.h"
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/mirrored_object_id.msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/object.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace vm {

class Instruction;
class MirroredObject;
class RwMutexedObject;

enum OperandAccessType {
  kConstOperandAccess = 0,
  kMutableOperandAccess,
  kDeleteOperandAccess,
};

// clang-format off
OBJECT_MSG_BEGIN(RwMutexedObjectAccess);
  // methods
  OF_PUBLIC void __Init__(Instruction* instruction, MirroredObject* mirrored_object,
                       OperandAccessType access_type);

  OF_PUBLIC bool is_const_operand() const;
  OF_PUBLIC bool is_mut_operand() const;
  OF_PUBLIC bool is_del_operand() const;

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(OperandAccessType, access_type);
  OBJECT_MSG_DEFINE_PTR(Instruction, instruction);
  OBJECT_MSG_DEFINE_PTR(MirroredObject, mirrored_object);
  OBJECT_MSG_DEFINE_PTR(RwMutexedObject, rw_mutexed_object);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(instruction_access_link);
  OBJECT_MSG_DEFINE_LIST_LINK(rw_mutexed_object_access_link);
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, MirroredObjectId, mirrored_object_id);
  
OBJECT_MSG_END(RwMutexedObjectAccess);

class LogicalObject;
OBJECT_MSG_BEGIN(RwMutexedObject);
  // methods

  OF_PUBLIC template<typename T> bool Has() const {
    return dynamic_cast<const T*>(&object()) != nullptr;
  }
  OF_PUBLIC template<typename T> Maybe<const T&> Get() const {
    const T* obj = dynamic_cast<const T*>(&object());
    CHECK_NOTNULL_OR_RETURN(obj)
      << "cast to " << typeid(T).name() << "failed. "
      << "type: " << (object_ptr() ? typeid(*object_ptr()).name() : "nullptr");
    return *obj;
  }
  OF_PUBLIC template<typename T> Maybe<T*> Mut() {
    T* obj = dynamic_cast<T*>(object_ptr().get());
    CHECK_NOTNULL_OR_RETURN(obj)
      << "cast to " << typeid(T).name() << "failed. "
      << "type: " << (object_ptr() ? typeid(*object_ptr()).name() : "nullptr");
    return obj;
  }
  OF_PUBLIC template<typename T, typename... Args> T* Init(Args&&... args) {
    T* object = dynamic_cast<T*>(object_ptr().get());
    CHECK(object == nullptr);
    object = new T(std::forward<Args>(args)...);
    reset_object(object);
    return object;
  }
  OF_PUBLIC const Object& object() const { return *object_ptr().get(); }
  OF_PUBLIC bool has_object() const { return object_ptr().get() != nullptr; }
  OF_PUBLIC void reset_object(Object* object) { mut_object_ptr()->reset(object); }
  OF_PUBLIC void reset_object() { reset_object(nullptr); }

  // fields
  OBJECT_MSG_DEFINE_STRUCT(std::unique_ptr<Object>, object_ptr);

  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(RwMutexedObjectAccess, rw_mutexed_object_access_link, access_list);
OBJECT_MSG_END(RwMutexedObject);

OBJECT_MSG_BEGIN(MirroredObject);
  // methods
  OF_PUBLIC void __Init__(LogicalObject* logical_object, int64_t global_device_id);

  //fields
  OBJECT_MSG_DEFINE_FLAT_MSG(MirroredObjectId, mirrored_object_id);
  OBJECT_MSG_DEFINE_OPTIONAL(RwMutexedObject, rw_mutexed_object);

  // links
  OBJECT_MSG_DEFINE_MAP_KEY(int64_t, global_device_id);
OBJECT_MSG_END(MirroredObject);

class VirtualMachine;
OBJECT_MSG_BEGIN(LogicalObject);
  // methods
  OF_PUBLIC void __Init__(const ObjectId& logical_object_id) {
    __Init__(logical_object_id, std::shared_ptr<ParallelDesc>());
  }
  OF_PUBLIC void __Init__(const ObjectId& logical_object_id,
                       const std::shared_ptr<ParallelDesc>& parallel_desc) {
    set_logical_object_id(logical_object_id);
    *mutable_parallel_desc() = parallel_desc;
  }

  // fields
  OBJECT_MSG_DEFINE_STRUCT(std::shared_ptr<ParallelDesc>, parallel_desc);

  // links
  OBJECT_MSG_DEFINE_MAP_KEY(ObjectId, logical_object_id);
  OBJECT_MSG_DEFINE_LIST_LINK(delete_link);
  // heads
  OBJECT_MSG_DEFINE_MAP_HEAD(MirroredObject, global_device_id, global_device_id2mirrored_object);
OBJECT_MSG_END(LogicalObject);
// clang-format on

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
