#ifndef ONEFLOW_CORE_COMMON_LIST_HEAD_H_
#define ONEFLOW_CORE_COMMON_LIST_HEAD_H_

#include "oneflow/core/common/struct_traits.h"
#include <glog/logging.h>

namespace oneflow {

struct ListHead final {
 public:
  ListHead() { Clear(); }

  ListHead* prev() const { return prev_; }
  ListHead* next() const { return next_; }

  void AppendTo(ListHead* prev) {
    prev->set_next(this);
    this->set_prev(prev);
  }
  void Clear() {
    prev_ = this;
    next_ = this;
  }

 private:
  void set_prev(ListHead* prev) { prev_ = prev; }
  void set_next(ListHead* next) { next_ = next; }

  ListHead* prev_;
  ListHead* next_;
};

#define EMBEDDED_LIST_VIEW(container_type, container_field) \
  EmbeddedListView<STRUCT_FIELD(container_type, container_field)>

#define DEFINE_EMBEDDED_LIST_VIEW(container_type, container_field, item_type, item_field) \
  DEFINE_STRUCT_FIELD(container_type, container_field)                                    \
  DEFINE_STRUCT_FIELD(item_type, item_field)                                              \
  _DEFINE_EMBEDDED_LIST_VIEW(STRUCT_FIELD(container_type, container_field),               \
                             STRUCT_FIELD(item_type, item_field))

template<typename ContainerField>
class EmbeddedListView final {};

template<typename ContainerField, typename ItemField>
class EmbeddedListViewImpl {
 public:
  using container_type = typename ContainerField::struct_type;
  using item_type = typename ItemField::struct_type;

  static_assert(std::is_same<typename ContainerField::field_type, ListHead>::value,
                "no ListHead found in container");
  static_assert(std::is_same<typename ItemField::field_type, ListHead>::value,
                "no ListHead found in item");
  explicit EmbeddedListViewImpl(container_type* container) : container_(container) {}
  ~EmbeddedListViewImpl() = default;

  void Erase(item_type* item) {
    CHECK_NE(item, end_item());
    ListHead* list_head = ItemField::FieldPtr4StructPtr(item);
    list_head->next()->AppendTo(list_head->prev());
    list_head->Clear();
  }
  void PushBack(item_type* item) { InsertAfter(last_item(), item); }
  void PushFront(item_type* item) { InsertAfter(end_item(), item); }
  item_type* PopBack() {
    CHECK(!empty());
    item_type* last = last_item();
    Erase(last);
    return last;
  }
  item_type* PopFront() {
    CHECK(!empty());
    item_type* first = begin_item();
    Erase(first);
    return first;
  }

  bool empty() const { return begin_item() == end_item(); }
  item_type* begin_item() const { return next_item(end_item()); }
  item_type* last_item() const { return prev_item(end_item()); }
  item_type* end_item() const {
    return ItemField::StructPtr4FieldPtr(ContainerField::FieldPtr4StructPtr(container_));
  }
  item_type* next_item(item_type* current) const {
    return ItemField::StructPtr4FieldPtr(ItemField::FieldPtr4StructPtr(current)->next());
  }
  item_type* prev_item(item_type* current) const {
    return ItemField::StructPtr4FieldPtr(ItemField::FieldPtr4StructPtr(current)->prev());
  }

 private:
  void InsertAfter(item_type* prev_item, item_type* new_item) {
    ListHead* prev_list_head = ItemField::FieldPtr4StructPtr(prev_item);
    ListHead* next_list_head = prev_list_head->next();
    ListHead* new_list_head = ItemField::FieldPtr4StructPtr(new_item);
    new_list_head->AppendTo(prev_list_head);
    next_list_head->AppendTo(new_list_head);
  }
  container_type* container_;
};

#define _DEFINE_EMBEDDED_LIST_VIEW(container_field, item_field)           \
  template<>                                                              \
  class EmbeddedListView<container_field> final                           \
      : public EmbeddedListViewImpl<container_field, item_field> {        \
   public:                                                                \
    explicit EmbeddedListView(container_type* container)                  \
        : EmbeddedListViewImpl<container_field, item_field>(container) {} \
  };

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_LIST_HEAD_H_
