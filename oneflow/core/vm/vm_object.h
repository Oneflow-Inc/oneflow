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
#ifndef ONEFLOW_CORE_VM_VM_OBJECT_H_
#define ONEFLOW_CORE_VM_VM_OBJECT_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/mirrored_object_id.h"
#include "oneflow/core/vm/stream_desc.h"
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

class RwMutexedObjectAccess final : public intrusive::Base {
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
  const intrusive::ListHook& rw_mutexed_object_access_hook() const {
    return rw_mutexed_object_access_hook_;
  }
  const MirroredObjectId& mirrored_object_id() const { return mirrored_object_id_.key().Get(); }
  bool is_mirrored_object_id_inserted() const { return !mirrored_object_id_.empty(); }

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
  MirroredObjectId* mut_mirrored_object_id() { return mirrored_object_id_.mut_key()->Mutable(); }

  // methods
  void __Init__(Instruction* instruction, MirroredObject* mirrored_object,
                OperandAccessType access_type);

  bool is_const_operand() const;
  bool is_mut_operand() const;

  intrusive::Ref::RefCntType ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }  // NOLINT

  RwMutexedObjectAccess()
      : intrusive_ref_(),
        access_type_(),
        instruction_(),
        mirrored_object_(),
        rw_mutexed_object_(),
        mirrored_object_id_(),
        instruction_access_hook_(),
        rw_mutexed_object_access_hook_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  OperandAccessType access_type_;
  Instruction* instruction_;
  MirroredObject* mirrored_object_;
  RwMutexedObject* rw_mutexed_object_;

 public:
  // skiplist hooks
  intrusive::SkipListHook<FlatMsg<MirroredObjectId>, 10> mirrored_object_id_;
  // list hooks
  intrusive::ListHook instruction_access_hook_;
  intrusive::ListHook rw_mutexed_object_access_hook_;
};  // NOLINT

struct LogicalObject;
class RwMutexedObject final : public intrusive::Base {
 public:
  void __Init__() {}
  // types
  using RwMutexedObjectAccessList =
      intrusive::List<INTRUSIVE_FIELD(RwMutexedObjectAccess, rw_mutexed_object_access_hook_)>;

  // Getters
  const RwMutexedObjectAccessList& access_list() const { return access_list_; }
  // Setters
  RwMutexedObjectAccessList* mut_access_list() { return &access_list_; }

  // methods
  template<typename T>
  bool Has() const {
    return dynamic_cast<const T*>(&object()) != nullptr;
  }
  template<typename T>
  Maybe<const T&> Get() const {
    const T* obj = dynamic_cast<const T*>(&object());
    const auto& origin_obj = *object_ptr_;
    CHECK_NOTNULL_OR_RETURN(obj) << "cast to " << typeid(T).name() << "failed. "
                                 << "type: "
                                 << (object_ptr_ ? typeid(origin_obj).name() : "nullptr");
    return *obj;
  }
  template<typename T>
  Maybe<T*> Mut() {
    T* obj = dynamic_cast<T*>(object_ptr_.get());
    const auto& origin_obj = *object_ptr_;
    CHECK_NOTNULL_OR_RETURN(obj) << "cast to " << typeid(T).name() << "failed. "
                                 << "type: "
                                 << (object_ptr_ ? typeid(origin_obj).name() : "nullptr");
    return obj;
  }
  template<typename T, typename... Args>
  T* Init(Args&&... args) {
    T* object = dynamic_cast<T*>(object_ptr_.get());
    CHECK(object == nullptr);
    object = new T(std::forward<Args>(args)...);
    reset_object(object);
    return object;
  }
  const Object& object() const { return *object_ptr_; }
  bool has_object() const { return static_cast<bool>(object_ptr_); }
  void reset_object(Object* object) { object_ptr_.reset(object); }
  void reset_object() { reset_object(nullptr); }

  intrusive::Ref::RefCntType ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  RwMutexedObject() : intrusive_ref_(), object_ptr_(), access_list_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  std::unique_ptr<Object> object_ptr_;

  // list hooks
  RwMutexedObjectAccessList access_list_;
};

class MirroredObject final : public intrusive::Base {
 public:
  // Getters
  bool has_deleting_access() const { return deleting_access_ != nullptr; }
  const RwMutexedObjectAccess& deleting_access() const { return *deleting_access_; }
  const RwMutexedObject& rw_mutexed_object() const {
    if (rw_mutexed_object_) { return rw_mutexed_object_.Get(); }
    static const auto default_val = intrusive::make_shared<RwMutexedObject>();
    return default_val.Get();
  }
  const MirroredObjectId& mirrored_object_id() const { return mirrored_object_id_.Get(); }
  int64_t global_device_id() const { return global_device_id_.key(); }
  // Setters
  void set_deleting_access(RwMutexedObjectAccess* val) { deleting_access_ = val; }
  void clear_deleting_access() { deleting_access_ = nullptr; }
  RwMutexedObjectAccess* mut_deleting_access() { return deleting_access_; }
  RwMutexedObject* mut_rw_mutexed_object() {
    if (!rw_mutexed_object_) { rw_mutexed_object_ = intrusive::make_shared<RwMutexedObject>(); }
    return rw_mutexed_object_.Mutable();
  }
  void reset_rw_mutexed_object(RwMutexedObject* rw_mutexed_object) {
    rw_mutexed_object_.Reset(rw_mutexed_object);
  }
  void reset_rw_mutexed_object(const RwMutexedObject& rw_mutexed_object) {
    rw_mutexed_object_.Reset(const_cast<RwMutexedObject*>(&rw_mutexed_object));
  }
  MirroredObjectId* mut_mirrored_object_id() { return mirrored_object_id_.Mutable(); }
  void set_global_device_id(int64_t val) { *global_device_id_.mut_key() = val; }

  // methods
  void __Init__() { clear_deleting_access(); }
  void __Init__(LogicalObject* logical_object, int64_t global_device_id);

  intrusive::Ref::RefCntType ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  MirroredObject()
      : intrusive_ref_(),
        mirrored_object_id_(),
        rw_mutexed_object_(),
        deleting_access_(),
        global_device_id_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  FlatMsg<MirroredObjectId> mirrored_object_id_;
  intrusive::shared_ptr<RwMutexedObject> rw_mutexed_object_;
  RwMutexedObjectAccess* deleting_access_;

 public:
  // skiplist hooks
  intrusive::SkipListHook<int64_t, 10> global_device_id_;
};

struct VirtualMachineEngine;
class LogicalObject final : public intrusive::Base {
 public:
  // types
  using GlobalDeviceId2MirroredObject =
      intrusive::SkipList<INTRUSIVE_FIELD(MirroredObject, global_device_id_)>;
  // Getters
  const std::shared_ptr<const ParallelDesc>& parallel_desc() const { return parallel_desc_; }
  const intrusive::ListHook& delete_hook() const { return delete_hook_; }
  const ObjectId& logical_object_id() const { return logical_object_id_.key(); }
  const GlobalDeviceId2MirroredObject& global_device_id2mirrored_object() const {
    return global_device_id2mirrored_object_;
  }
  // Setters
  std::shared_ptr<const ParallelDesc>* mut_parallel_desc() { return &parallel_desc_; }
  void set_logical_object_id(const ObjectId& val) { *logical_object_id_.mut_key() = val; }
  GlobalDeviceId2MirroredObject* mut_global_device_id2mirrored_object() {
    return &global_device_id2mirrored_object_;
  }

  // methods
  void __Init__() { /* Do nothing */
  }
  void __Init__(const ObjectId& logical_object_id) {
    __Init__(logical_object_id, std::shared_ptr<const ParallelDesc>());
  }
  void __Init__(const ObjectId& logical_object_id,
                const std::shared_ptr<const ParallelDesc>& parallel_desc) {
    set_logical_object_id(logical_object_id);
    *mut_parallel_desc() = parallel_desc;
  }

  intrusive::Ref::RefCntType ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  LogicalObject()
      : intrusive_ref_(),
        parallel_desc_(),
        global_device_id2mirrored_object_(),
        logical_object_id_(),
        delete_hook_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  std::shared_ptr<const ParallelDesc> parallel_desc_;
  // maps
  GlobalDeviceId2MirroredObject global_device_id2mirrored_object_;

 public:
  // skiplist hooks
  intrusive::SkipListHook<ObjectId, 24> logical_object_id_;
  // list hooks
  intrusive::ListHook delete_hook_;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_OBJECT_H_
