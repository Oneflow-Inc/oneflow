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
#ifndef ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_LIST_H_
#define ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_LIST_H_

#include <typeinfo>
#include "oneflow/core/object_msg/object_msg_core.h"
#include "oneflow/core/object_msg/embedded_list.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_LIST_LINK(field_name)                                     \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                \
  _OBJECT_MSG_DEFINE_LIST_LINK(STATIC_COUNTER(field_counter), field_name);

#define OBJECT_MSG_DEFINE_LIST_HEAD(elem_type, elem_field_name, field_name)               \
  static_assert(__is_object_message_type__, "this struct is not a object message");       \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                      \
  _OBJECT_MSG_DEFINE_LIST_HEAD(STATIC_COUNTER(field_counter), elem_type, elem_field_name, \
                               field_name);

#define OBJECT_MSG_LIST(obj_msg_type, obj_msg_field) \
  ObjectMsgList<_OBJECT_MSG_LIST_STRUCT_FIELD(obj_msg_type, obj_msg_field)>

#define OBJECT_MSG_LIST_PTR(obj_msg_type, obj_msg_field) \
  TrivialObjectMsgList<kDisableSelfLoopLink,             \
                       _OBJECT_MSG_LIST_STRUCT_FIELD(obj_msg_type, obj_msg_field)>*

#define OBJECT_MSG_LIST_FOR_EACH(list_ptr, elem) \
  _OBJECT_MSG_LIST_FOR_EACH(std::remove_pointer<decltype(list_ptr)>::type, list_ptr, elem)

#define OBJECT_MSG_LIST_FOR_EACH_PTR(list_ptr, elem) \
  _OBJECT_MSG_LIST_FOR_EACH_PTR(std::remove_pointer<decltype(list_ptr)>::type, list_ptr, elem)

#define OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(list_ptr, elem)                                     \
  _OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(std::remove_pointer<decltype(list_ptr)>::type, list_ptr, \
                                       elem)

// details

#define _OBJECT_MSG_LIST_STRUCT_FIELD(obj_msg_type, obj_msg_field)   \
  StructField<OBJECT_MSG_TYPE_CHECK(obj_msg_type), EmbeddedListLink, \
              OBJECT_MSG_TYPE_CHECK(obj_msg_type)::OF_PP_CAT(obj_msg_field, _kDssFieldOffset)>

#define _OBJECT_MSG_DEFINE_LIST_HEAD(field_counter, elem_type, elem_field_name, field_name)    \
  _OBJECT_MSG_DEFINE_LIST_HEAD_FIELD_TYPE(elem_type, elem_field_name, field_name)              \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _ObjectMsgListType), \
                   OF_PP_CAT(field_name, _));                                                  \
  _OBJECT_MSG_DEFINE_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name)                   \
  OBJECT_MSG_DEFINE_LIST_ELEM_STRUCT(field_counter, elem_type, elem_field_name, field_name);   \
  OBJECT_MSG_DEFINE_LIST_LINK_EDGES(field_counter, elem_type, elem_field_name, field_name);    \
  OBJECT_MSG_OVERLOAD_INIT_WITH_FIELD_INDEX(field_counter, ObjectMsgEmbeddedListHeadInit);     \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedListHeadDelete);

#define _OBJECT_MSG_DEFINE_LIST_HEAD_FIELD_TYPE(elem_type, elem_field_name, field_name)            \
 public:                                                                                           \
  using OF_PP_CAT(field_name, _ObjectMsgListType) =                                                \
      TrivialObjectMsgList<GetObjectMsgLinkType<std::is_same<self_type, elem_type>::value>::value, \
                           StructField<OBJECT_MSG_TYPE_CHECK(elem_type), EmbeddedListLink,         \
                                       OBJECT_MSG_TYPE_CHECK(elem_type)::OF_PP_CAT(                \
                                           elem_field_name, _kDssFieldOffset)>>;

#define _OBJECT_MSG_DEFINE_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name) \
 public:                                                                           \
  const OF_PP_CAT(field_name, _ObjectMsgListType) & field_name() const {           \
    return OF_PP_CAT(field_name, _);                                               \
  }                                                                                \
  OF_PP_CAT(field_name, _ObjectMsgListType) * OF_PP_CAT(mut_, field_name)() {      \
    return &OF_PP_CAT(field_name, _);                                              \
  }                                                                                \
  OF_PP_CAT(field_name, _ObjectMsgListType) * OF_PP_CAT(mutable_, field_name)() {  \
    return &OF_PP_CAT(field_name, _);                                              \
  }                                                                                \
                                                                                   \
 private:                                                                          \
  OF_PP_CAT(field_name, _ObjectMsgListType) OF_PP_CAT(field_name, _);

#define OBJECT_MSG_DEFINE_LIST_ELEM_STRUCT(field_counter, elem_type, elem_field_name, field_name) \
 public:                                                                                          \
  template<typename Enabled>                                                                      \
  struct ContainerElemStruct<field_counter, Enabled> final {                                      \
    using type = elem_type;                                                                       \
  };

#define OBJECT_MSG_DEFINE_LIST_LINK_EDGES(field_counter, elem_type, elem_field_name, field_name) \
 public:                                                                                         \
  template<typename Enable>                                                                      \
  struct LinkEdgesGetter<field_counter, Enable> final {                                          \
    static void Call(std::set<ObjectMsgContainerLinkEdge>* edges) {                              \
      ObjectMsgContainerLinkEdge edge;                                                           \
      edge.container_type_name = typeid(self_type).name();                                       \
      edge.container_field_name = OF_PP_STRINGIZE(field_name) "_";                               \
      edge.elem_type_name = typeid(elem_type).name();                                            \
      edge.elem_link_name = OF_PP_STRINGIZE(elem_field_name) "_";                                \
      edges->insert(edge);                                                                       \
    }                                                                                            \
  };

#define _OBJECT_MSG_DEFINE_LIST_LINK(field_counter, field_name)               \
  _OBJECT_MSG_DEFINE_LIST_LINK_FIELD(field_name)                              \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedListLinkInit);     \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedListLinkDelete); \
  DSS_DEFINE_FIELD(field_counter, "object message", EmbeddedListLink, OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_LIST_LINK_FIELD(field_name)         \
 public:                                                       \
  bool OF_PP_CAT(is_, OF_PP_CAT(field_name, _empty))() const { \
    return OF_PP_CAT(field_name, _).empty();                   \
  }                                                            \
                                                               \
 private:                                                      \
  EmbeddedListLink OF_PP_CAT(field_name, _);

#define _OBJECT_MSG_LIST_FOR_EACH(list_type, list_ptr, elem)                          \
  for (ObjectMsgPtr<typename list_type::value_type> elem, *end_if_not_null = nullptr; \
       end_if_not_null == nullptr; end_if_not_null = nullptr, ++end_if_not_null)      \
  EMBEDDED_LIST_FOR_EACH_WITH_EXPR(                                                   \
      (StructField<typename list_type, EmbeddedListLink,                              \
                   list_type::ContainerLinkOffset()>::FieldPtr4StructPtr(list_ptr)),  \
      list_type::value_link_struct_field, elem_ptr, (elem.Reset(elem_ptr), true))

#define _OBJECT_MSG_LIST_FOR_EACH_PTR(list_type, list_ptr, elem)                     \
  EMBEDDED_LIST_FOR_EACH(                                                            \
      (StructField<typename list_type, EmbeddedListLink,                             \
                   list_type::ContainerLinkOffset()>::FieldPtr4StructPtr(list_ptr)), \
      list_type::value_link_struct_field, elem)

#define _OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(list_type, list_ptr, elem)              \
  EMBEDDED_LIST_UNSAFE_FOR_EACH(                                                     \
      (StructField<typename list_type, EmbeddedListLink,                             \
                   list_type::ContainerLinkOffset()>::FieldPtr4StructPtr(list_ptr)), \
      list_type::value_link_struct_field, elem)

template<int field_index, typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedListHeadInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    field->template __Init__<field_index>();
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedListHeadDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) { field->Clear(); }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedListLinkInit {
  static void Call(WalkCtxType* ctx, EmbeddedListLink* field) { field->__Init__(); }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedListLinkDelete {
  static void Call(WalkCtxType* ctx, EmbeddedListLink* field) { CHECK(field->empty()); }
};

enum ObjectMsgLinkType { kDisableSelfLoopLink = 0, kEnableSelfLoopLink };

template<bool enable_self_loop_link>
struct GetObjectMsgLinkType {};
template<>
struct GetObjectMsgLinkType<true> {
  static const ObjectMsgLinkType value = kEnableSelfLoopLink;
};
template<>
struct GetObjectMsgLinkType<false> {
  static const ObjectMsgLinkType value = kDisableSelfLoopLink;
};

template<ObjectMsgLinkType link_type, typename ValueLinkField>
class TrivialObjectMsgList;

template<typename ValueLinkField>
class TrivialObjectMsgList<kDisableSelfLoopLink, ValueLinkField> {
 public:
  using value_type = typename ValueLinkField::struct_type;
  using value_link_struct_field = ValueLinkField;

  template<typename Enabled = void>
  static constexpr int ContainerLinkOffset() {
    return offsetof(TrivialObjectMsgList, list_head_)
           + EmbeddedListHead<ValueLinkField>::ContainerLinkOffset();
  }

  std::size_t size() const { return list_head_.size(); }
  bool empty() const { return list_head_.empty(); }

  void CheckSize() const { list_head_.CheckSize(); }

  template<int field_index = 0>
  void __Init__() {
    list_head_.__Init__();
  }

  value_type* Begin() {
    if (list_head_.empty()) { return nullptr; }
    return list_head_.Begin();
  }
  value_type* Next(value_type* ptr) {
    if (ptr == nullptr) { return nullptr; }
    value_type* next = list_head_.Next(ptr);
    if (next == list_head_.End()) { return nullptr; }
    return next;
  }
  value_type* Last() {
    if (list_head_.empty()) { return nullptr; }
    return list_head_.Last();
  }
  constexpr value_type* End() { return nullptr; }

  void MoveToDstBack(value_type* ptr, TrivialObjectMsgList* dst) {
    list_head_.MoveToDstBack(ptr, &dst->list_head_);
  }
  void MoveToDstFront(value_type* ptr, TrivialObjectMsgList* dst) {
    list_head_.MoveToDstFront(ptr, &dst->list_head_);
  }
  value_type* MoveFrontToDstBack(TrivialObjectMsgList* dst) {
    value_type* begin = list_head_.Begin();
    MoveToDstBack(begin, dst);
    return begin;
  }
  value_type* MoveBackToDstBack(TrivialObjectMsgList* dst) {
    value_type* begin = list_head_.Last();
    MoveToDstBack(begin, dst);
    return begin;
  }

  void PushBack(value_type* ptr) {
    list_head_.PushBack(ptr);
    ObjectMsgPtrUtil::Ref(ptr);
  }

  void PushFront(value_type* ptr) {
    list_head_.PushFront(ptr);
    ObjectMsgPtrUtil::Ref(ptr);
  }

  void EmplaceBack(ObjectMsgPtr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    ptr.__UnsafeMoveTo__(&raw_ptr);
    list_head_.PushBack(raw_ptr);
  }

  void EmplaceFront(ObjectMsgPtr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    ptr.__UnsafeMoveTo__(&raw_ptr);
    list_head_.PushFront(raw_ptr);
  }

  ObjectMsgPtr<value_type> Erase(value_type* ptr) {
    list_head_.Erase(ptr);
    return ObjectMsgPtr<value_type>::__UnsafeMove__(ptr);
  }

  ObjectMsgPtr<value_type> PopBack() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopBack(); }
    return ObjectMsgPtr<value_type>::__UnsafeMove__(raw_ptr);
  }

  ObjectMsgPtr<value_type> PopFront() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopFront(); }
    return ObjectMsgPtr<value_type>::__UnsafeMove__(raw_ptr);
  }

  void MoveTo(TrivialObjectMsgList* list) { MoveToDstBack(list); }
  void MoveToDstBack(TrivialObjectMsgList* list) { list_head_.MoveToDstBack(&list->list_head_); }

  void Clear() {
    while (!empty()) { ObjectMsgPtrUtil::ReleaseRef(list_head_.PopFront()); }
  }

 private:
  EmbeddedListHead<ValueLinkField> list_head_;
};

template<typename ValueLinkField>
class TrivialObjectMsgList<kEnableSelfLoopLink, ValueLinkField> {
 public:
  using value_type = typename ValueLinkField::struct_type;
  using value_link_struct_field = ValueLinkField;

  template<typename Enabled = void>
  static constexpr int ContainerLinkOffset() {
    return offsetof(TrivialObjectMsgList, list_head_)
           + EmbeddedListHead<ValueLinkField>::ContainerLinkOffset();
  }

  std::size_t size() const { return list_head_.size(); }
  bool empty() const { return list_head_.empty(); }

  template<int field_index>
  void __Init__() {
    list_head_.__Init__();
    using ThisInContainer =
        StructField<value_type, TrivialObjectMsgList,
                    value_type::template __DssFieldOffset4FieldIndex__<field_index>::value>;
    container_ = ThisInContainer::StructPtr4FieldPtr(this);
  }

  value_type* Begin() {
    if (list_head_.empty()) { return nullptr; }
    return list_head_.Begin();
  }
  value_type* Next(value_type* ptr) {
    if (ptr == nullptr) { return nullptr; }
    value_type* next = list_head_.Next(ptr);
    if (next == list_head_.End()) { return nullptr; }
    return next;
  }
  value_type* Last() {
    if (list_head_.empty()) { return nullptr; }
    return list_head_.Last();
  }
  constexpr value_type* End() { return nullptr; }

  void MoveToDstBack(value_type* ptr, TrivialObjectMsgList* dst) {
    list_head_.MoveToDstBack(ptr, &dst->list_head_);
    MoveReference(ptr, dst);
  }
  void MoveToDstFront(value_type* ptr, TrivialObjectMsgList* dst) {
    list_head_.MoveToDstFront(ptr, &dst->list_head_);
    MoveReference(ptr, dst);
  }
  value_type* MoveFrontToDstBack(TrivialObjectMsgList* dst) {
    value_type* begin = list_head_.Begin();
    MoveToDstBack(begin, dst);
    return begin;
  }
  value_type* MoveBackToDstBack(TrivialObjectMsgList* dst) {
    value_type* begin = list_head_.Last();
    MoveToDstBack(begin, dst);
    return begin;
  }

  void PushBack(value_type* ptr) {
    list_head_.PushBack(ptr);
    if (container_ != ptr) { ObjectMsgPtrUtil::Ref(ptr); }
  }

  void PushFront(value_type* ptr) {
    list_head_.PushFront(ptr);
    if (container_ != ptr) { ObjectMsgPtrUtil::Ref(ptr); }
  }

  void EmplaceBack(ObjectMsgPtr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    if (container_ != ptr.Mutable()) {
      ptr.__UnsafeMoveTo__(&raw_ptr);
    } else {
      raw_ptr = ptr.Mutable();
    }
    list_head_.PushBack(raw_ptr);
  }

  void EmplaceFront(ObjectMsgPtr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    if (container_ != ptr.Mutable()) {
      ptr.__UnsafeMoveTo__(&raw_ptr);
    } else {
      raw_ptr = ptr.Mutable();
    }
    list_head_.PushFront(raw_ptr);
  }

  ObjectMsgPtr<value_type> Erase(value_type* ptr) {
    list_head_.Erase(ptr);
    if (container_ != ptr) {
      return ObjectMsgPtr<value_type>::__UnsafeMove__(ptr);
    } else {
      return ObjectMsgPtr<value_type>(ptr);
    }
  }

  ObjectMsgPtr<value_type> PopBack() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopBack(); }
    if (container_ != raw_ptr) {
      return ObjectMsgPtr<value_type>::__UnsafeMove__(raw_ptr);
    } else {
      return ObjectMsgPtr<value_type>(raw_ptr);
    }
  }

  ObjectMsgPtr<value_type> PopFront() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopFront(); }
    if (container_ != raw_ptr) {
      return ObjectMsgPtr<value_type>::__UnsafeMove__(raw_ptr);
    } else {
      return ObjectMsgPtr<value_type>(raw_ptr);
    }
  }

  void MoveTo(TrivialObjectMsgList* list) { MoveToDstBack(list); }
  void MoveToDstBack(TrivialObjectMsgList* list) {
    while (!empty()) { MoveToDstBack(list_head_.Begin(), list); }
  }

  void Clear() {
    while (!empty()) {
      auto* ptr = list_head_.PopFront();
      if (container_ != ptr) { ObjectMsgPtrUtil::ReleaseRef(ptr); }
    }
  }

 private:
  void MoveReference(value_type* ptr, TrivialObjectMsgList* dst) {
    if (ptr == container_ && ptr != dst->container_) {
      ObjectMsgPtrUtil::Ref(ptr);
    } else if (ptr != container_ && ptr == dst->container_) {
      ObjectMsgPtrUtil::ReleaseRef(ptr);
    } else {
      // do nothing
    }
  }

  EmbeddedListHead<ValueLinkField> list_head_;
  const value_type* container_;
};

template<typename LinkField>
class ObjectMsgList : public TrivialObjectMsgList<kDisableSelfLoopLink, LinkField> {
 public:
  ObjectMsgList(const ObjectMsgList&) = delete;
  ObjectMsgList(ObjectMsgList&&) = delete;
  ObjectMsgList() { this->__Init__(); }
  ~ObjectMsgList() { this->Clear(); }
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_OBJECT_MSG_LIST_H_
