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

struct Instruction;
struct MirroredObject;
struct RwMutexedObject;

enum OperandAccessType {
  kConstOperandAccess = 0,
  kMutableOperandAccess,
};

// clang-format off
OBJECT_MSG_BEGIN(RwMutexedObjectAccess);
 public:
  void __Init__();
  // Getters
  OperandAccessType access_type() const { return access_type_; }
  bool has_instruction() const { return instruction_ != nullptr; }
  bool has_mirrored_object() const { return mirrored_object_ != nullptr; }
  bool has_rw_mutexed_object() const { return rw_mutexed_object_ != nullptr; }
  const Instruction& instruction() const { return *instruction_; }
  const MirroredObject& mirrored_object() const { return *mirrored_object_; }
  const RwMutexedObject& rw_mutexed_object() const { return *rw_mutexed_object_; }
  bool is_rw_mutexed_object_access_entry_empty() const { return rw_mutexed_object_access_entry_.empty(); }

  // Setters
  void set_access_type(OperandAccessType val) { access_type_ = val; }
  void set_instruction(Instruction* val) { instruction_ = val; }
  void set_mirrored_object(MirroredObject* val) { mirrored_object_ = val; }
  void set_rw_mutexed_object(RwMutexedObject* val) { rw_mutexed_object_ = val; }
  void clear_instruction() { instruction_ = nullptr; }
  void clear_mirrored_object() { mirrored_object_ = nullptr; }
  void clear_rw_mutexed_object() { rw_mutexed_object_ = nullptr; }
  Instruction* mut_instruction() { return instruction_; }
  MirroredObject* mut_mirrored_object() { return mirrored_object_; }
  RwMutexedObject* mut_rw_mutexed_object() { return rw_mutexed_object_; }
  Instruction* mutable_instruction() { return instruction_; }
  MirroredObject* mutable_mirrored_object() { return mirrored_object_; }
  RwMutexedObject* mutable_rw_mutexed_object() { return rw_mutexed_object_; }

  // methods
  OF_PUBLIC void __Init__(Instruction* instruction, MirroredObject* mirrored_object,
                       OperandAccessType access_type);

  OF_PUBLIC bool is_const_operand() const;
  OF_PUBLIC bool is_mut_operand() const;

  // fields
  OBJECT_MSG_DEFINE_FIELD(OperandAccessType, access_type_);
  OBJECT_MSG_DEFINE_FIELD(Instruction*, instruction_);
  OBJECT_MSG_DEFINE_FIELD(MirroredObject*, mirrored_object_);
  OBJECT_MSG_DEFINE_FIELD(RwMutexedObject*, rw_mutexed_object_);

  // list entries
  OBJECT_MSG_DEFINE_FIELD(intrusive::ListEntry, instruction_access_entry_);
  OBJECT_MSG_DEFINE_FIELD(intrusive::ListEntry, rw_mutexed_object_access_entry_);
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, MirroredObjectId, mirrored_object_id);
  
OBJECT_MSG_END(RwMutexedObjectAccess);

struct LogicalObject;
OBJECT_MSG_BEGIN(RwMutexedObject);
 public:
  void __Init__() {}

  // types
  using RwMutexedObjectAccessList = intrusive::List<OBJECT_MSG_FIELD(RwMutexedObjectAccess, rw_mutexed_object_access_entry_)>;

  // Getters
  const RwMutexedObjectAccessList& access_list() const { return access_list_; }
  // Setters
  RwMutexedObjectAccessList* mut_access_list() { return &access_list_; }
  RwMutexedObjectAccessList* mutable_access_list() { return &access_list_; }

  // methods
  OF_PUBLIC template<typename T> bool Has() const {
    return dynamic_cast<const T*>(&object()) != nullptr;
  }
  OF_PUBLIC template<typename T> Maybe<const T&> Get() const {
    const T* obj = dynamic_cast<const T*>(&object());
    const auto &origin_obj = *object_ptr_;
    CHECK_NOTNULL_OR_RETURN(obj)
      << "cast to " << typeid(T).name() << "failed. "
      << "type: " << (object_ptr_ ? typeid(origin_obj).name() : "nullptr");
    return *obj;
  }
  OF_PUBLIC template<typename T> Maybe<T*> Mut() {
    T* obj = dynamic_cast<T*>(object_ptr_.get());
    const auto &origin_obj = *object_ptr_;
    CHECK_NOTNULL_OR_RETURN(obj)
      << "cast to " << typeid(T).name() << "failed. "
      << "type: " << (object_ptr_ ? typeid(origin_obj).name() : "nullptr");
    return obj;
  }
  OF_PUBLIC template<typename T, typename... Args> T* Init(Args&&... args) {
    T* object = dynamic_cast<T*>(object_ptr_.get());
    CHECK(object == nullptr);
    object = new T(std::forward<Args>(args)...);
    reset_object(object);
    return object;
  }
  OF_PUBLIC const Object& object() const { return *object_ptr_; }
  OF_PUBLIC bool has_object() const { return static_cast<bool>(object_ptr_); }
  OF_PUBLIC void reset_object(Object* object) { object_ptr_.reset(object); }
  OF_PUBLIC void reset_object() { reset_object(nullptr); }

  // fields
  OBJECT_MSG_DEFINE_FIELD(std::unique_ptr<Object>, object_ptr_);

  // list entries
  OBJECT_MSG_DEFINE_FIELD(RwMutexedObjectAccessList, access_list_);
OBJECT_MSG_END(RwMutexedObject);

OBJECT_MSG_BEGIN(MirroredObject);
 public:
  // Getters
  bool has_deleting_access() const { return deleting_access_ != nullptr; }
  const RwMutexedObjectAccess& deleting_access() const { return *deleting_access_; }
  const RwMutexedObject& rw_mutexed_object() const {
    if (rw_mutexed_object_) { return rw_mutexed_object_.Get(); }
    static const auto default_val = ObjectMsgPtr<RwMutexedObject>::New();
    return default_val.Get();
  }
  const MirroredObjectId& mirrored_object_id() const { return mirrored_object_id_.Get(); }
  // Setters
  void set_deleting_access(RwMutexedObjectAccess* val) { deleting_access_ = val; }
  void clear_deleting_access() { deleting_access_ = nullptr; }
  RwMutexedObjectAccess* mut_deleting_access() { return deleting_access_; }
  RwMutexedObjectAccess* mutable_deleting_access() { return deleting_access_; }
  RwMutexedObject* mut_rw_mutexed_object() { return mutable_rw_mutexed_object(); }
  RwMutexedObject* mutable_rw_mutexed_object() {
    if (!rw_mutexed_object_) { rw_mutexed_object_ = ObjectMsgPtr<RwMutexedObject>::New(); }
    return rw_mutexed_object_.Mutable();
  }
  void reset_rw_mutexed_object(RwMutexedObject* rw_mutexed_object) {
    rw_mutexed_object_.Reset(rw_mutexed_object);
  }
  void reset_rw_mutexed_object(const RwMutexedObject& rw_mutexed_object) {
    rw_mutexed_object_.Reset(const_cast<RwMutexedObject*>(&rw_mutexed_object));
  }
  MirroredObjectId* mut_mirrored_object_id() { return mirrored_object_id_.Mutable(); }
  MirroredObjectId* mutable_mirrored_object_id() { return mirrored_object_id_.Mutable(); }


  // methods
  OF_PUBLIC void __Init__() { clear_deleting_access(); }
  OF_PUBLIC void __Init__(LogicalObject* logical_object, int64_t global_device_id);

  //fields
  OBJECT_MSG_DEFINE_FIELD(FlatMsg<MirroredObjectId>, mirrored_object_id_);
  OBJECT_MSG_DEFINE_FIELD(ObjectMsgPtr<RwMutexedObject>, rw_mutexed_object_);
  OBJECT_MSG_DEFINE_FIELD(RwMutexedObjectAccess*, deleting_access_);


  // list entries
  OBJECT_MSG_DEFINE_MAP_KEY(int64_t, global_device_id);
OBJECT_MSG_END(MirroredObject);

struct VirtualMachine;
OBJECT_MSG_BEGIN(LogicalObject);
 public:
  LogicalObject() = default;
  // Getters
  const std::shared_ptr<const ParallelDesc>& parallel_desc() const { return parallel_desc_; }
  bool is_delete_entry_empty() const { return delete_entry_.empty(); }
  // Setters
  std::shared_ptr<const ParallelDesc>* mut_parallel_desc() { return &parallel_desc_; }
  std::shared_ptr<const ParallelDesc>* mutable_parallel_desc() { return &parallel_desc_; }

  // methods
  OF_PUBLIC void __Init__() { /* Do nothing */ }
  OF_PUBLIC void __Init__(const ObjectId& logical_object_id) {
    __Init__(logical_object_id, std::shared_ptr<const ParallelDesc>());
  }
  OF_PUBLIC void __Init__(const ObjectId& logical_object_id,
                       const std::shared_ptr<const ParallelDesc>& parallel_desc) {
    set_logical_object_id(logical_object_id);
    *mutable_parallel_desc() = parallel_desc;
  }

  // fields
  OBJECT_MSG_DEFINE_FIELD(std::shared_ptr<const ParallelDesc>, parallel_desc_);

  // list entries
  OBJECT_MSG_DEFINE_MAP_KEY(ObjectId, logical_object_id);
  OBJECT_MSG_DEFINE_FIELD(intrusive::ListEntry, delete_entry_);
  // heads
  OBJECT_MSG_DEFINE_MAP_HEAD(MirroredObject, global_device_id, global_device_id2mirrored_object);
OBJECT_MSG_END(LogicalObject);
// clang-format on

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
