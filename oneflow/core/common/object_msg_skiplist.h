#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_SKIPLIST_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_SKIPLIST_H_

#include "oneflow/core/common/object_msg_core.h"
#include "oneflow/core/common/embedded_skiplist.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_SKIPLIST_KEY(max_level, T, field_name)                    \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  _OBJECT_MSG_DEFINE_SKIPLIST_KEY(DSS_GET_FIELD_COUNTER(), max_level, T, field_name);

#define OBJECT_MSG_DEFINE_SKIPLIST_HEAD(field_type, field_name)                     \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  _OBJECT_MSG_DEFINE_SKIPLIST_HEAD(DSS_GET_FIELD_COUNTER(), field_type, field_name);

// details

#define _OBJECT_MSG_DEFINE_SKIPLIST_HEAD(field_counter, field_type, field_name)   \
  _OBJECT_MSG_DEFINE_SKIPLIST_HEAD_FIELD(field_type, field_name)                  \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedSkipListHeadInit);     \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedSkipListHeadDelete); \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_SKIPLIST_HEAD_FIELD(field_type, field_name)                         \
 public:                                                                                       \
  using OF_PP_CAT(field_name, _ObjectMsgSkipListType) =                                        \
      TrivialObjectMsgSkipList<OBJECT_MSG_SKIPLIST_ELEM_STRUCT_FIELD(field_type, field_name)>; \
  const OF_PP_CAT(field_name, _ObjectMsgSkipListType) & field_name() const {                   \
    return OF_PP_CAT(field_name, _);                                                           \
  }                                                                                            \
  OF_PP_CAT(field_name, _ObjectMsgSkipListType) * OF_PP_CAT(mut_, field_name)() {              \
    return &OF_PP_CAT(field_name, _);                                                          \
  }                                                                                            \
  OF_PP_CAT(field_name, _ObjectMsgSkipListType) * OF_PP_CAT(mutable_, field_name)() {          \
    return &OF_PP_CAT(field_name, _);                                                          \
  }                                                                                            \
                                                                                               \
 private:                                                                                      \
  OF_PP_CAT(field_name, _ObjectMsgSkipListType) OF_PP_CAT(field_name, _);

#define _OBJECT_MSG_DEFINE_SKIPLIST_KEY(field_counter, max_level, T, field_name)      \
  _OBJECT_MSG_DEFINE_SKIPLIST_KEY_FIELD(max_level, T, field_name)                     \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedSkipListIteratorInit);     \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedSkipListIteratorDelete); \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_SKIPLIST_KEY_FIELD(max_level, key_type, field_name)                     \
 public:                                                                                           \
  using OF_PP_CAT(field_name, _ObjectMsgSkipListKeyType) =                                         \
      EmbeddedSkipListKey<key_type, max_level>;                                                    \
  const key_type& OF_PP_CAT(field_name, _key)() const { return OF_PP_CAT(field_name, _).key(); }   \
  key_type* OF_PP_CAT(OF_PP_CAT(mut_, field_name), _key)() {                                       \
    return OF_PP_CAT(field_name, _).mut_key();                                                     \
  }                                                                                                \
  template<typename T>                                                                             \
  void OF_PP_CAT(OF_PP_CAT(set_, field_name), _key)(const T& val) {                                \
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value, "T is not scalar type"); \
    *OF_PP_CAT(OF_PP_CAT(mut_, field_name), _key)() = val;                                         \
  }                                                                                                \
                                                                                                   \
 private:                                                                                          \
  EmbeddedSkipListKey<key_type, max_level> OF_PP_CAT(field_name, _);

#define OBJECT_MSG_SKIPLIST(obj_msg_type, obj_msg_field) \
  ObjectMsgSkipList<OBJECT_MSG_SKIPLIST_ELEM_STRUCT_FIELD(obj_msg_type, obj_msg_field)>

#define OBJECT_MSG_SKIPLIST_ELEM_STRUCT_FIELD(field_type, field_name)                        \
  StructField<OBJECT_MSG_TYPE(field_type),                                                   \
              OBJECT_MSG_TYPE(field_type)::OF_PP_CAT(field_name, _ObjectMsgSkipListKeyType), \
              OBJECT_MSG_TYPE(field_type)::OF_PP_CAT(field_name, _DssFieldOffset)()>

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedSkipListHeadInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    field->__Init__();
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedSkipListHeadDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    field->Clear();
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedSkipListIteratorInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    field->__Init__();
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedSkipListIteratorDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    field->CheckEmpty();
  }
};

template<typename ElemKeyField>
class TrivialObjectMsgSkipList {
 public:
  using elem_type = typename ElemKeyField::struct_type;
  using key_type = typename ElemKeyField::field_type::key_type;

  void __Init__() { skiplist_head_.__Init__(); }

  std::size_t size() const { return skiplist_head_.size(); }
  bool empty() const { return skiplist_head_.empty(); }

  ObjectMsgPtr<elem_type> Find(const key_type& key) {
    ObjectMsgPtr<elem_type> ret;
    ret.Reset(skiplist_head_.Find(key));
    return ret;
  }
  bool EqualsEnd(const ObjectMsgPtr<elem_type>& ptr) { return !ptr; }
  void Erase(const key_type& key) { ObjectMsgPtrUtil::ReleaseRef(skiplist_head_.Erase(key)); }
  void Erase(ObjectMsgPtr<elem_type>* elem_ptr) {
    skiplist_head_.Erase(elem_ptr->Mutable());
    ObjectMsgPtrUtil::ReleaseRef(elem_ptr->Mutable());
  }
  std::pair<ObjectMsgPtr<elem_type>, bool> Insert(ObjectMsgPtr<elem_type>* elem_ptr) {
    elem_type* ret_elem = nullptr;
    bool success = false;
    std::tie(ret_elem, success) = skiplist_head_.Insert(elem_ptr->Mutable());
    std::pair<ObjectMsgPtr<elem_type>, bool> ret;
    ret.first.Reset(ret_elem);
    ret.second = success;
    if (success) { ObjectMsgPtrUtil::Ref(elem_ptr->Mutable()); }
    return ret;
  }

  void Clear() {
    skiplist_head_.Clear([](elem_type* elem) { ObjectMsgPtrUtil::ReleaseRef(elem); });
  }

 private:
  EmbeddedSkipListHead<ElemKeyField> skiplist_head_;
};

template<typename ItemField>
class ObjectMsgSkipList final : public TrivialObjectMsgSkipList<ItemField> {
 public:
  ObjectMsgSkipList() { this->__Init__(); }
  ~ObjectMsgSkipList() { this->Clear(); }
};
}

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_SKIPLIST_H_
