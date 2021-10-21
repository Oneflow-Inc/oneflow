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
#ifndef ONEFLOW_CORE_OBJECT_MSG_CONDITION_LIST_H_
#define ONEFLOW_CORE_OBJECT_MSG_CONDITION_LIST_H_

#include <mutex>
#include <condition_variable>
#include "oneflow/core/object_msg/object_msg_list.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(elem_type, elem_field_name, field_name)           \
  static_assert(__is_object_message_type__, "this struct is not a object message");             \
  static_assert(!std::is_same<self_type, elem_type>::value, "self loop link is not supported"); \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                            \
  _OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(STATIC_COUNTER(field_counter), elem_type,              \
                                         elem_field_name, field_name);

#define OBJECT_MSG_CONDITION_LIST(obj_msg_type, obj_msg_field)                              \
  ObjectMsgConditionList<StructField<OBJECT_MSG_TYPE_CHECK(obj_msg_type), EmbeddedListLink, \
                                     OBJECT_MSG_TYPE_CHECK(obj_msg_type)::OF_PP_CAT(        \
                                         obj_msg_field, _kDssFieldOffset)>>

// details

#define _OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(field_counter, elem_type, elem_field_name,      \
                                               field_name)                                     \
  _OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name)         \
  OBJECT_MSG_DEFINE_CONDITION_LIST_ELEM_STRUCT(field_counter, elem_type, elem_field_name,      \
                                               field_name);                                    \
  OBJECT_MSG_DEFINE_CONDITION_LIST_LINK_EDGES(field_counter, elem_type, elem_field_name,       \
                                              field_name);                                     \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedConditionListHeadInit);             \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedConditionListHeadDelete);         \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _ObjectMsgListType), \
                   OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name)   \
 public:                                                                                       \
  using OF_PP_CAT(field_name, _ObjectMsgListType) = TrivialObjectMsgConditionList<StructField< \
      OBJECT_MSG_TYPE_CHECK(elem_type), EmbeddedListLink,                                      \
      OBJECT_MSG_TYPE_CHECK(elem_type)::OF_PP_CAT(elem_field_name, _kDssFieldOffset)>>;        \
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

#define OBJECT_MSG_DEFINE_CONDITION_LIST_ELEM_STRUCT(field_counter, elem_type, elem_field_name, \
                                                     field_name)                                \
 public:                                                                                        \
  template<typename Enabled>                                                                    \
  struct ContainerElemStruct<field_counter, Enabled> final {                                    \
    using type = elem_type;                                                                     \
  };

#define OBJECT_MSG_DEFINE_CONDITION_LIST_LINK_EDGES(field_counter, elem_type, elem_field_name, \
                                                    field_name)                                \
 public:                                                                                       \
  template<typename Enable>                                                                    \
  struct LinkEdgesGetter<field_counter, Enable> final {                                        \
    static void Call(std::set<ObjectMsgContainerLinkEdge>* edges) {                            \
      ObjectMsgContainerLinkEdge edge;                                                         \
      edge.container_type_name = typeid(self_type).name();                                     \
      edge.container_field_name = OF_PP_STRINGIZE(field_name) "_";                             \
      edge.elem_type_name = typeid(elem_type).name();                                          \
      edge.elem_link_name = OF_PP_STRINGIZE(elem_field_name) "_";                              \
      edges->insert(edge);                                                                     \
    }                                                                                          \
  };

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedConditionListHeadInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) { field->__Init__(); }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedConditionListHeadDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) { field->__Delete__(); }
};

enum ObjectMsgConditionListStatus {
  kObjectMsgConditionListStatusSuccess = 0,
  kObjectMsgConditionListStatusErrorClosed,
};

template<typename LinkField>
class TrivialObjectMsgConditionList {
 public:
  using value_type = typename LinkField::struct_type;

  void __Init__() {
    list_head_.__Init__();
    is_closed_ = false;
    new (mutex_buff_) std::mutex();
    new (cond_buff_) std::condition_variable();
  }

  bool Empty() {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    return list_head_.empty();
  }

  ObjectMsgConditionListStatus EmplaceBack(ObjectMsgPtr<value_type>&& ptr) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    if (is_closed_) { return kObjectMsgConditionListStatusErrorClosed; }
    list_head_.EmplaceBack(std::move(ptr));
    mut_cond()->notify_one();
    return kObjectMsgConditionListStatusSuccess;
  }
  ObjectMsgConditionListStatus PushBack(value_type* ptr) {
    return EmplaceBack(ObjectMsgPtr<value_type>(ptr));
  }
  ObjectMsgConditionListStatus PopFront(ObjectMsgPtr<value_type>* ptr) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    mut_cond()->wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kObjectMsgConditionListStatusErrorClosed; }
    *ptr = list_head_.PopFront();
    return kObjectMsgConditionListStatusSuccess;
  }

  ObjectMsgConditionListStatus MoveFrom(
      TrivialObjectMsgList<kDisableSelfLoopLink, LinkField>* src) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    if (is_closed_) { return kObjectMsgConditionListStatusErrorClosed; }
    src->MoveToDstBack(&list_head_);
    mut_cond()->notify_one();
    return kObjectMsgConditionListStatusSuccess;
  }

  ObjectMsgConditionListStatus MoveTo(TrivialObjectMsgList<kDisableSelfLoopLink, LinkField>* dst) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    mut_cond()->wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kObjectMsgConditionListStatusErrorClosed; }
    list_head_.MoveToDstBack(dst);
    return kObjectMsgConditionListStatusSuccess;
  }

  ObjectMsgConditionListStatus TryMoveTo(
      TrivialObjectMsgList<kDisableSelfLoopLink, LinkField>* dst) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    if (list_head_.empty()) { return kObjectMsgConditionListStatusSuccess; }
    mut_cond()->wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kObjectMsgConditionListStatusErrorClosed; }
    list_head_.MoveToDstBack(dst);
    return kObjectMsgConditionListStatusSuccess;
  }

  void Close() {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    is_closed_ = true;
    mut_cond()->notify_all();
  }

  void __Delete__() {
    list_head_.Clear();
    using namespace std;
    mut_mutex()->mutex::~mutex();
    mut_cond()->condition_variable::~condition_variable();
  }

 private:
  std::mutex* mut_mutex() { return reinterpret_cast<std::mutex*>(&mutex_buff_[0]); }
  std::condition_variable* mut_cond() {
    return reinterpret_cast<std::condition_variable*>(&cond_buff_[0]);
  }

  TrivialObjectMsgList<kDisableSelfLoopLink, LinkField> list_head_;
  union {
    char mutex_buff_[sizeof(std::mutex)];
    int64_t mutex_buff_align_;
  };
  union {
    char cond_buff_[sizeof(std::condition_variable)];
    int64_t cond_buff_align_;
  };
  bool is_closed_;
};

template<typename LinkField>
class ObjectMsgConditionList : public TrivialObjectMsgConditionList<LinkField> {
 public:
  ObjectMsgConditionList(const ObjectMsgConditionList&) = delete;
  ObjectMsgConditionList(ObjectMsgConditionList&&) = delete;
  ObjectMsgConditionList() { this->__Init__(); }
  ~ObjectMsgConditionList() { this->__Delete__(); }
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_CONDITION_LIST_H_
