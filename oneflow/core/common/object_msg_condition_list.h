#ifndef ONEFLOW_CORE_COMMON_CONDITION_LIST_H_
#define ONEFLOW_CORE_COMMON_CONDITION_LIST_H_

#include <mutex>
#include <condition_variable>
#include "oneflow/core/common/object_msg_list.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(elem_type, elem_field_name, field_name)         \
  static_assert(__is_object_message_type__, "this struct is not a object message");           \
  _OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(DSS_GET_FIELD_COUNTER(), elem_type, elem_field_name, \
                                         field_name);

#define OBJECT_MSG_CONDITION_LIST(obj_msg_type, obj_msg_field)                              \
  ObjectMsgConditionList<StructField<OBJECT_MSG_TYPE_CHECK(obj_msg_type), EmbeddedListLink, \
                                     OBJECT_MSG_TYPE_CHECK(obj_msg_type)::OF_PP_CAT(        \
                                         obj_msg_field, _DssFieldOffset)()>>

// details

#define _OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(field_counter, elem_type, elem_field_name, \
                                               field_name)                                \
  _OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name)    \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedConditionListHeadInit);        \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedConditionListHeadDelete);    \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name)   \
 public:                                                                                       \
  using OF_PP_CAT(field_name, _ObjectMsgListType) = TrivialObjectMsgConditionList<StructField< \
      OBJECT_MSG_TYPE_CHECK(elem_type), EmbeddedListLink,                                      \
      OBJECT_MSG_TYPE_CHECK(elem_type)::OF_PP_CAT(elem_field_name, _DssFieldOffset)()>>;       \
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

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedConditionListHeadInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    field->__Init__();
  }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedConditionListHeadDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    field->Clear();
  }
};

enum ObjectMsgConditionListStatus {
  kObjectMsgConditionListStatusSuccess = 0,
  kObjectMsgConditionListStatusErrorClosed
};

template<typename LinkField>
class TrivialObjectMsgConditionList {
 public:
  using value_type = typename LinkField::struct_type;

  void __Init__() {
    list_head_.__Init__();
    is_closed_ = false;
  }

  ObjectMsgConditionListStatus EmplaceBack(ObjectMsgPtr<value_type>&& ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (is_closed_) { return kObjectMsgConditionListStatusErrorClosed; }
    list_head_.EmplaceBack(std::move(ptr));
    cond_.notify_one();
    return kObjectMsgConditionListStatusSuccess;
  }
  ObjectMsgConditionListStatus PopFront(ObjectMsgPtr<value_type>* ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kObjectMsgConditionListStatusErrorClosed; }
    *ptr = list_head_.PopFront();
    return kObjectMsgConditionListStatusSuccess;
  }

  ObjectMsgConditionListStatus MoveFrom(TrivialObjectMsgList<LinkField>* src) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (is_closed_) { return kObjectMsgConditionListStatusErrorClosed; }
    src->MoveToDstBack(&list_head_);
    cond_.notify_one();
    return kObjectMsgConditionListStatusSuccess;
  }

  ObjectMsgConditionListStatus MoveTo(TrivialObjectMsgList<LinkField>* dst) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kObjectMsgConditionListStatusErrorClosed; }
    list_head_.MoveToDstBack(dst);
    return kObjectMsgConditionListStatusSuccess;
  }

  void Close() {
    std::unique_lock<std::mutex> lock(mutex_);
    is_closed_ = true;
    cond_.notify_all();
  }

  void Clear() { list_head_.Clear(); }

 private:
  TrivialObjectMsgList<LinkField> list_head_;
  std::mutex mutex_;
  std::condition_variable cond_;
  bool is_closed_;
};

template<typename LinkField>
class ObjectMsgConditionList : public TrivialObjectMsgConditionList<LinkField> {
 public:
  ObjectMsgConditionList(const ObjectMsgConditionList&) = delete;
  ObjectMsgConditionList(ObjectMsgConditionList&&) = delete;
  ObjectMsgConditionList() { this->__Init__(); }
  ~ObjectMsgConditionList() { this->Clear(); }
};
}

#endif  // ONEFLOW_CORE_COMMON_CONDITION_LIST_H_
