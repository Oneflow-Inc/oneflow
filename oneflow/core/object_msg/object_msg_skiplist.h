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
#ifndef ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_SKIPLIST_H_
#define ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_SKIPLIST_H_

#include "oneflow/core/object_msg/object_msg_core.h"
#include "oneflow/core/object_msg/embedded_skiplist.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_SKIPLIST_KEY(max_level, T, field_name)                          \
  static_assert(__is_object_message_type__, "this struct is not a object message");       \
  static_assert(std::is_standard_layout<T>::value, "this struct is not standard layout"); \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                      \
  _OBJECT_MSG_DEFINE_SKIPLIST_KEY(STATIC_COUNTER(field_counter), max_level, T, field_name);

#define OBJECT_MSG_DEFINE_SKIPLIST_HEAD(elem_type, elem_field_name, field_name)                 \
  static_assert(__is_object_message_type__, "this struct is not a object message");             \
  static_assert(!std::is_same<self_type, elem_type>::value, "self loop link is not supported"); \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                            \
  _OBJECT_MSG_DEFINE_SKIPLIST_HEAD(STATIC_COUNTER(field_counter), elem_type, elem_field_name,   \
                                   field_name);

#define OBJECT_MSG_SKIPLIST(elem_type, elem_field_name) \
  ObjectMsgSkipList<OBJECT_MSG_SKIPLIST_ELEM_STRUCT_FIELD(elem_type, elem_field_name)>

#define OBJECT_MSG_SKIPLIST_FOR_EACH(skiplist_ptr, elem)                                         \
  _OBJECT_MSG_SKIPLIST_FOR_EACH(std::remove_pointer<decltype(skiplist_ptr)>::type, skiplist_ptr, \
                                elem)

#define OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(skiplist_ptr, elem)                           \
  _OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(std::remove_pointer<decltype(skiplist_ptr)>::type, \
                                    skiplist_ptr, elem)

#define OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(skiplist_ptr, elem)                           \
  _OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(std::remove_pointer<decltype(skiplist_ptr)>::type, \
                                           skiplist_ptr, elem)
// details

#define _OBJECT_MSG_DEFINE_SKIPLIST_HEAD(field_counter, elem_type, elem_field_name, field_name)    \
  _OBJECT_MSG_DEFINE_SKIPLIST_HEAD_FIELD(elem_type, elem_field_name, field_name);                  \
  OBJECT_MSG_DEFINE_SKIPLIST_ELEM_STRUCT(field_counter, elem_type, elem_field_name, field_name);   \
  OBJECT_MSG_DEFINE_SKIPLIST_LINK_EDGES(field_counter, elem_type, elem_field_name, field_name);    \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedSkipListHeadInit);                      \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedSkipListHeadDelete);                  \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _ObjectMsgSkipListType), \
                   OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_SKIPLIST_HEAD_FIELD(elem_type, elem_field_name, field_name)             \
 public:                                                                                           \
  using OF_PP_CAT(field_name, _ObjectMsgSkipListType) =                                            \
      TrivialObjectMsgSkipList<OBJECT_MSG_SKIPLIST_ELEM_STRUCT_FIELD(elem_type, elem_field_name)>; \
  const OF_PP_CAT(field_name, _ObjectMsgSkipListType) & field_name() const {                       \
    return OF_PP_CAT(field_name, _);                                                               \
  }                                                                                                \
  OF_PP_CAT(field_name, _ObjectMsgSkipListType) * OF_PP_CAT(mut_, field_name)() {                  \
    return &OF_PP_CAT(field_name, _);                                                              \
  }                                                                                                \
  OF_PP_CAT(field_name, _ObjectMsgSkipListType) * OF_PP_CAT(mutable_, field_name)() {              \
    return &OF_PP_CAT(field_name, _);                                                              \
  }                                                                                                \
                                                                                                   \
 private:                                                                                          \
  OF_PP_CAT(field_name, _ObjectMsgSkipListType) OF_PP_CAT(field_name, _);

#define OBJECT_MSG_DEFINE_SKIPLIST_ELEM_STRUCT(field_counter, elem_type, elem_field_name, \
                                               field_name)                                \
 public:                                                                                  \
  template<typename Enabled>                                                              \
  struct ContainerElemStruct<field_counter, Enabled> final {                              \
    using type = elem_type;                                                               \
  };

#define OBJECT_MSG_DEFINE_SKIPLIST_LINK_EDGES(field_counter, elem_type, elem_field_name, \
                                              field_name)                                \
 public:                                                                                 \
  template<typename Enable>                                                              \
  struct LinkEdgesGetter<field_counter, Enable> final {                                  \
    static void Call(std::set<ObjectMsgContainerLinkEdge>* edges) {                      \
      ObjectMsgContainerLinkEdge edge;                                                   \
      edge.container_type_name = typeid(self_type).name();                               \
      edge.container_field_name = OF_PP_STRINGIZE(field_name) "_";                       \
      edge.elem_type_name = typeid(elem_type).name();                                    \
      edge.elem_link_name = OF_PP_STRINGIZE(elem_field_name) "_";                        \
      edges->insert(edge);                                                               \
    }                                                                                    \
  };

#define _OBJECT_MSG_DEFINE_SKIPLIST_KEY(field_counter, max_level, T, field_name)      \
  _OBJECT_MSG_DEFINE_SKIPLIST_KEY_FIELD(max_level, T, field_name)                     \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedSkipListIteratorInit);     \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedSkipListIteratorDelete); \
  DSS_DEFINE_FIELD(field_counter, "object message",                                   \
                   OF_PP_CAT(field_name, _ObjectMsgSkipListKeyType), OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_SKIPLIST_KEY_FIELD(max_level, key_type, field_name)               \
 public:                                                                                     \
  using OF_PP_CAT(field_name, _ObjectMsgSkipListKeyType) =                                   \
      EmbeddedSkipListKey<key_type, max_level>;                                              \
  bool OF_PP_CAT(is_, OF_PP_CAT(field_name, _inserted))() const {                            \
    return !OF_PP_CAT(field_name, _).empty();                                                \
  }                                                                                          \
  ConstType<key_type>& field_name() const { return OF_PP_CAT(field_name, _).key(); }         \
  key_type* OF_PP_CAT(mut_, field_name)() { return OF_PP_CAT(field_name, _).mut_key(); }     \
  key_type* OF_PP_CAT(mutable_, field_name)() { return OF_PP_CAT(field_name, _).mut_key(); } \
  template<typename T>                                                                       \
  void OF_PP_CAT(set_, field_name)(const T& val) {                                           \
    static_assert(std::is_scalar<T>::value, "T is not scalar type");                         \
    *OF_PP_CAT(mut_, field_name)() = val;                                                    \
  }                                                                                          \
                                                                                             \
 private:                                                                                    \
  OF_PP_CAT(field_name, _ObjectMsgSkipListKeyType) OF_PP_CAT(field_name, _);

#define OBJECT_MSG_SKIPLIST_ELEM_STRUCT_FIELD(elem_type, elem_field_name)                \
  StructField<elem_type,                                                                 \
              typename elem_type::OF_PP_CAT(elem_field_name, _ObjectMsgSkipListKeyType), \
              elem_type::OF_PP_CAT(elem_field_name, _kDssFieldOffset)>

#define _OBJECT_MSG_SKIPLIST_FOR_EACH(skiplist_type, skiplist_ptr, elem)                     \
  for (ObjectMsgPtr<skiplist_type::value_type> elem, *end_if_not_null = nullptr;             \
       end_if_not_null == nullptr; end_if_not_null = nullptr, ++end_if_not_null)             \
  EMBEDDED_LIST_FOR_EACH_WITH_EXPR(                                                          \
      (StructField<                                                                          \
          skiplist_type, EmbeddedListLink,                                                   \
          skiplist_type::ContainerLevelZeroLinkOffset()>::FieldPtr4StructPtr(skiplist_ptr)), \
      skiplist_type::elem_level0_link_struct_field, elem_ptr, (elem.Reset(elem_ptr), true))

#define _OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(skiplist_type, skiplist_ptr, elem)                 \
  EMBEDDED_LIST_FOR_EACH(                                                                    \
      (StructField<                                                                          \
          skiplist_type, EmbeddedListLink,                                                   \
          skiplist_type::ContainerLevelZeroLinkOffset()>::FieldPtr4StructPtr(skiplist_ptr)), \
      skiplist_type::elem_level0_link_struct_field, elem)

#define _OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(skiplist_type, skiplist_ptr, elem)          \
  EMBEDDED_LIST_UNSAFE_FOR_EACH(                                                             \
      (StructField<                                                                          \
          skiplist_type, EmbeddedListLink,                                                   \
          skiplist_type::ContainerLevelZeroLinkOffset()>::FieldPtr4StructPtr(skiplist_ptr)), \
      skiplist_type::elem_level0_link_struct_field, elem)

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedSkipListHeadInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) { field->__Init__(); }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedSkipListHeadDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) { field->Clear(); }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedSkipListIteratorInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) { field->__Init__(); }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedSkipListIteratorDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) { field->CheckEmpty(); }
};

template<typename ElemKeyField>
class TrivialObjectMsgSkipList {
 public:
  using value_type = typename ElemKeyField::struct_type;
  using key_type = typename ElemKeyField::field_type::key_type;
  using elem_key_level0_link_struct_field =
      StructField<typename ElemKeyField::field_type, EmbeddedListLink,
                  ElemKeyField::field_type::LevelZeroLinkOffset()>;
  using elem_level0_link_struct_field =
      typename ComposeStructField<ElemKeyField, elem_key_level0_link_struct_field>::type;
  template<typename Enabled = void>
  static constexpr int ContainerLevelZeroLinkOffset() {
    return offsetof(TrivialObjectMsgSkipList, skiplist_head_)
           + EmbeddedSkipListHead<ElemKeyField>::ContainerLevelZeroLinkOffset();
  }

  void __Init__() { skiplist_head_.__Init__(); }

  std::size_t size() const { return skiplist_head_.size(); }
  bool empty() const { return skiplist_head_.empty(); }
  value_type* Begin() { return skiplist_head_.Begin(); }
  ObjectMsgPtr<value_type> Find(const key_type& key) {
    ObjectMsgPtr<value_type> ret;
    ret.Reset(skiplist_head_.Find(key));
    return ret;
  }
  value_type* FindPtr(const key_type& key) { return skiplist_head_.Find(key); }
  const value_type* FindPtr(const key_type& key) const { return skiplist_head_.Find(key); }
  bool EqualsEnd(const ObjectMsgPtr<value_type>& ptr) { return !ptr; }
  void Erase(const key_type& key) { ObjectMsgPtrUtil::ReleaseRef(skiplist_head_.Erase(key)); }
  void Erase(value_type* elem_ptr) {
    skiplist_head_.Erase(elem_ptr);
    ObjectMsgPtrUtil::ReleaseRef(elem_ptr);
  }
  std::pair<ObjectMsgPtr<value_type>, bool> Insert(value_type* elem_ptr) {
    value_type* ret_elem = nullptr;
    bool success = false;
    std::tie(ret_elem, success) = skiplist_head_.Insert(elem_ptr);
    std::pair<ObjectMsgPtr<value_type>, bool> ret;
    ret.first.Reset(ret_elem);
    ret.second = success;
    if (success) { ObjectMsgPtrUtil::Ref(elem_ptr); }
    return ret;
  }

  void Clear() {
    skiplist_head_.Clear([](value_type* elem) { ObjectMsgPtrUtil::ReleaseRef(elem); });
  }

 private:
  EmbeddedSkipListHead<ElemKeyField> skiplist_head_;
};

template<typename ItemField>
class ObjectMsgSkipList final : public TrivialObjectMsgSkipList<ItemField> {
 public:
  ObjectMsgSkipList(const ObjectMsgSkipList&) = delete;
  ObjectMsgSkipList(ObjectMsgSkipList&&) = delete;

  ObjectMsgSkipList() { this->__Init__(); }
  ~ObjectMsgSkipList() { this->Clear(); }
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_SKIPLIST_H_
