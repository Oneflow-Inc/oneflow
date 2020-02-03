#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_LIST_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_LIST_H_

#include "oneflow/core/common/object_msg_core.h"
#include "oneflow/core/common/embedded_list.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_LIST_LINK(field_name)                                     \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  _OBJECT_MSG_DEFINE_LIST_LINK(DSS_GET_FIELD_COUNTER(), field_name);

#define OBJECT_MSG_DEFINE_LIST_HEAD(elem_type, elem_field_name, field_name)         \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  _OBJECT_MSG_DEFINE_LIST_HEAD(DSS_GET_FIELD_COUNTER(), elem_type, elem_field_name, field_name);

#define OBJECT_MSG_LIST_FOR_EACH(list_ptr, elem)                            \
  for (ObjectMsgPtr<std::remove_pointer<decltype((list_ptr)->End())>::type> \
           elem = (list_ptr)->Begin(),                                      \
           __next_elem__ = (list_ptr)->Next(elem.Mutable());                \
       elem.Mutable() != (list_ptr)->End();                                 \
       elem = __next_elem__, __next_elem__ = (list_ptr)->Next(__next_elem__.Mutable()))

#define OBJECT_MSG_LIST_FOR_EACH_UNSAFE(list_ptr, elem) \
  for (auto* elem = (list_ptr)->Begin(); elem != (list_ptr)->End(); elem = (list_ptr)->Next(elem))

// details

#define _OBJECT_MSG_DEFINE_LIST_HEAD(field_counter, elem_type, elem_field_name, field_name) \
  _OBJECT_MSG_DEFINE_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name)                \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedListHeadInit);                   \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedListHeadDelete);               \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name)             \
 public:                                                                                       \
  using OF_PP_CAT(field_name, _ObjectMsgListType) = TrivialObjectMsgList<                      \
      StructField<OBJECT_MSG_TYPE(elem_type), EmbeddedListLink,                                \
                  OBJECT_MSG_TYPE(elem_type)::OF_PP_CAT(elem_field_name, _DssFieldOffset)()>>; \
  const OF_PP_CAT(field_name, _ObjectMsgListType) & field_name() const {                       \
    return OF_PP_CAT(field_name, _);                                                           \
  }                                                                                            \
  OF_PP_CAT(field_name, _ObjectMsgListType) * OF_PP_CAT(mut_, field_name)() {                  \
    return &OF_PP_CAT(field_name, _);                                                          \
  }                                                                                            \
  OF_PP_CAT(field_name, _ObjectMsgListType) * OF_PP_CAT(mutable_, field_name)() {              \
    return &OF_PP_CAT(field_name, _);                                                          \
  }                                                                                            \
                                                                                               \
 private:                                                                                      \
  OF_PP_CAT(field_name, _ObjectMsgListType) OF_PP_CAT(field_name, _);

#define _OBJECT_MSG_DEFINE_LIST_LINK(field_counter, field_name)               \
  _OBJECT_MSG_DEFINE_LIST_LINK_FIELD(field_name)                              \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedListLinkInit);     \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedListLinkDelete); \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_LIST_LINK_FIELD(field_name) \
 private:                                              \
  EmbeddedListLink OF_PP_CAT(field_name, _);

#define OBJECT_MSG_LIST(obj_msg_type, obj_msg_field)               \
  ObjectMsgList<                                                   \
      StructField<OBJECT_MSG_TYPE(obj_msg_type), EmbeddedListLink, \
                  OBJECT_MSG_TYPE(obj_msg_type)::OF_PP_CAT(obj_msg_field, _DssFieldOffset)()>>

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedListHeadInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    field->__Init__();
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedListHeadDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    field->Clear();
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedListLinkInit {
  static void Call(WalkCtxType* ctx, EmbeddedListLink* field, const char* field_name) {
    field->__Init__();
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedListLinkDelete {
  static void Call(WalkCtxType* ctx, EmbeddedListLink* field, const char* field_name) {
    CHECK(field->empty());
  }
};

template<typename LinkField>
class TrivialObjectMsgList {
 public:
  using value_type = typename LinkField::struct_type;

  std::size_t size() const { return list_head_.size(); }
  bool empty() const { return list_head_.empty(); }

  void __Init__() { list_head_.__Init__(); }

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

  void PushBack(value_type* ptr) {
    list_head_.PushBack(ptr);
    ObjectMsgPtrUtil::Ref(ptr);
  }

  void PushFront(value_type* ptr) {
    list_head_.PushFront(ptr);
    ObjectMsgPtrUtil::Ref(ptr);
  }

  void Erase(value_type* ptr) {
    list_head_.Erase(ptr);
    ObjectMsgPtrUtil::ReleaseRef(ptr);
  }

  ObjectMsgPtr<value_type> PopBack() {
    ObjectMsgPtr<value_type> ptr;
    if (list_head_.empty()) { return ptr; }
    ptr.Reset(list_head_.PopBack());
    ObjectMsgPtrUtil::ReleaseRef(ptr.Mutable());
    return ptr;
  }

  ObjectMsgPtr<value_type> PopFront() {
    ObjectMsgPtr<value_type> ptr;
    if (list_head_.empty()) { return ptr; }
    ptr.Reset(list_head_.PopFront());
    ObjectMsgPtrUtil::ReleaseRef(ptr.Mutable());
    return ptr;
  }

  void MoveTo(TrivialObjectMsgList* list) { MoveToDstBack(list); }
  void MoveToDstBack(TrivialObjectMsgList* list) { list_head_.MoveToDstBack(&list->list_head_); }

  void Clear() {
    while (!empty()) { ObjectMsgPtrUtil::ReleaseRef(list_head_.PopFront()); }
  }

 private:
  EmbeddedListHead<LinkField> list_head_;
};

template<typename LinkField>
class ObjectMsgList : public TrivialObjectMsgList<LinkField> {
 public:
  ObjectMsgList(const ObjectMsgList&) = delete;
  ObjectMsgList(ObjectMsgList&&) = delete;
  ObjectMsgList() { this->__Init__(); }
  ~ObjectMsgList() { this->Clear(); }
};
}

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_LIST_H_
