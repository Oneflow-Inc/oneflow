#ifndef ONEFLOW_CORE_COMMON_MUTEXED_LIST_H_
#define ONEFLOW_CORE_COMMON_MUTEXED_LIST_H_

#include <mutex>
#include "oneflow/core/common/object_msg_list.h"

namespace oneflow {

#define OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD(elem_type, elem_field_name, field_name)               \
  static_assert(__is_object_message_type__, "this struct is not a object message");               \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                                 \
  _OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD(STATIC_COUNTER(field_counter), elem_type, elem_field_name, \
                                       field_name);

#define OBJECT_MSG_MUTEXED_LIST(obj_msg_type, obj_msg_field)                              \
  ObjectMsgMutexedList<StructField<OBJECT_MSG_TYPE_CHECK(obj_msg_type), EmbeddedListLink, \
                                   OBJECT_MSG_TYPE_CHECK(obj_msg_type)::OF_PP_CAT(        \
                                       obj_msg_field, _DssFieldOffset)()>>

// details

#define _OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD(field_counter, elem_type, elem_field_name, \
                                             field_name)                                \
  _OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name)    \
  OBJECT_MSG_DEFINE_MUTEXED_LIST_ELEM_STRUCT(field_counter, elem_type, elem_field_name, \
                                             field_name);                               \
  OBJECT_MSG_DEFINE_MUTEXED_LIST_LINK_EDGES(field_counter, elem_type, elem_field_name,  \
                                            field_name);                                \
  OBJECT_MSG_OVERLOAD_INIT(field_counter, ObjectMsgEmbeddedMutexedListHeadInit);        \
  OBJECT_MSG_OVERLOAD_DELETE(field_counter, ObjectMsgEmbeddedMutexedListHeadDelete);    \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(field_name, _));

#define _OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD_FIELD(elem_type, elem_field_name, field_name)        \
 public:                                                                                          \
  using OF_PP_CAT(field_name, _ObjectMsgListType) =                                               \
      TrivialObjectMsgMutexedList<StructField<OBJECT_MSG_TYPE_CHECK(elem_type), EmbeddedListLink, \
                                              OBJECT_MSG_TYPE_CHECK(elem_type)::OF_PP_CAT(        \
                                                  elem_field_name, _DssFieldOffset)()>>;          \
  const OF_PP_CAT(field_name, _ObjectMsgListType) & field_name() const {                          \
    return OF_PP_CAT(field_name, _);                                                              \
  }                                                                                               \
  OF_PP_CAT(field_name, _ObjectMsgListType) * OF_PP_CAT(mut_, field_name)() {                     \
    return &OF_PP_CAT(field_name, _);                                                             \
  }                                                                                               \
  OF_PP_CAT(field_name, _ObjectMsgListType) * OF_PP_CAT(mutable_, field_name)() {                 \
    return &OF_PP_CAT(field_name, _);                                                             \
  }                                                                                               \
                                                                                                  \
 private:                                                                                         \
  OF_PP_CAT(field_name, _ObjectMsgListType) OF_PP_CAT(field_name, _);

#define OBJECT_MSG_DEFINE_MUTEXED_LIST_ELEM_STRUCT(field_counter, elem_type, elem_field_name, \
                                                   field_name)                                \
 public:                                                                                      \
  template<typename Enabled>                                                                  \
  struct ContainerElemStruct<field_counter, Enabled> final {                                  \
    using type = elem_type;                                                                   \
  };

#define OBJECT_MSG_DEFINE_MUTEXED_LIST_LINK_EDGES(field_counter, elem_type, elem_field_name, \
                                                  field_name)                                \
 public:                                                                                     \
  template<typename Enable>                                                                  \
  struct LinkEdgesGetter<field_counter, Enable> final {                                      \
    static void Call(std::set<ObjectMsgContainerLinkEdge>* edges) {                          \
      ObjectMsgContainerLinkEdge edge;                                                       \
      edge.container_type_name = typeid(self_type).name();                                   \
      edge.container_field_name = OF_PP_STRINGIZE(field_name) "_";                           \
      edge.elem_type_name = typeid(elem_type).name();                                        \
      edge.elem_link_name = OF_PP_STRINGIZE(elem_field_name) "_";                            \
      edges->insert(edge);                                                                   \
    }                                                                                        \
  };

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedMutexedListHeadInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) { field->__Init__(); }
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgEmbeddedMutexedListHeadDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field) { field->Clear(); }
};

template<typename LinkField>
class TrivialObjectMsgMutexedList {
 public:
  using value_type = typename LinkField::struct_type;

  std::size_t size() const { return list_head_.size(); }
  bool empty() const { return list_head_.empty(); }

  void __Init__() { list_head_.__Init__(); }

  void EmplaceBack(ObjectMsgPtr<value_type>&& ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    return list_head_.EmplaceBack(std::move(ptr));
  }
  void EmplaceFront(ObjectMsgPtr<value_type>&& ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    return list_head_.EmplaceFront(std::move(ptr));
  }
  ObjectMsgPtr<value_type> PopBack() {
    std::unique_lock<std::mutex> lock(mutex_);
    return list_head_.PopBack();
  }
  ObjectMsgPtr<value_type> PopFront() {
    std::unique_lock<std::mutex> lock(mutex_);
    return list_head_.PopFront();
  }

  void MoveFrom(TrivialObjectMsgList<LinkField>* src) {
    std::unique_lock<std::mutex> lock(mutex_);
    src->MoveToDstBack(&list_head_);
  }

  void MoveTo(TrivialObjectMsgList<LinkField>* dst) {
    std::unique_lock<std::mutex> lock(mutex_);
    list_head_.MoveToDstBack(dst);
  }

  void Clear() { list_head_.Clear(); }

 private:
  TrivialObjectMsgList<LinkField> list_head_;
  std::mutex mutex_;
};

template<typename LinkField>
class ObjectMsgMutexedList : public TrivialObjectMsgMutexedList<LinkField> {
 public:
  ObjectMsgMutexedList(const ObjectMsgMutexedList&) = delete;
  ObjectMsgMutexedList(ObjectMsgMutexedList&&) = delete;
  ObjectMsgMutexedList() { this->__Init__(); }
  ~ObjectMsgMutexedList() { this->Clear(); }
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_MUTEXED_LIST_H_
