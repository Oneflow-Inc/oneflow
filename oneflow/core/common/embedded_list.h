#ifndef ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_
#define ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_

#include "oneflow/core/common/struct_traits.h"
#include <glog/logging.h>

namespace oneflow {

struct EmbeddedListItem final {
 public:
  EmbeddedListItem() { Clear(); }

  EmbeddedListItem* prev() const { return prev_; }
  EmbeddedListItem* next() const { return next_; }

  void AppendTo(EmbeddedListItem* prev) {
    prev->set_next(this);
    this->set_prev(prev);
  }
  void Clear() {
    prev_ = this;
    next_ = this;
  }

 private:
  void set_prev(EmbeddedListItem* prev) { prev_ = prev; }
  void set_next(EmbeddedListItem* next) { next_ = next; }

  EmbeddedListItem* prev_;
  EmbeddedListItem* next_;
};

template<typename ItemField>
class EmbeddedListHead {
 public:
  using item_type = typename ItemField::struct_type;
  static_assert(std::is_same<typename ItemField::field_type, EmbeddedListItem>::value,
                "no EmbeddedListItem found in item");

  void Clear() { container_.Clear(); }

  void Erase(item_type* item) {
    CHECK_NE(item, end_item());
    EmbeddedListItem* list_item = ItemField::FieldPtr4StructPtr(item);
    list_item->next()->AppendTo(list_item->prev());
    list_item->Clear();
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
  item_type* end_item() const { return ItemField::StructPtr4FieldPtr(&container_); }
  item_type* next_item(item_type* current) const {
    return ItemField::StructPtr4FieldPtr(ItemField::FieldPtr4StructPtr(current)->next());
  }
  item_type* prev_item(item_type* current) const {
    return ItemField::StructPtr4FieldPtr(ItemField::FieldPtr4StructPtr(current)->prev());
  }

 private:
  void InsertAfter(item_type* prev_item, item_type* new_item) {
    EmbeddedListItem* prev_list_item = ItemField::FieldPtr4StructPtr(prev_item);
    EmbeddedListItem* next_list_item = prev_list_item->next();
    EmbeddedListItem* new_list_item = ItemField::FieldPtr4StructPtr(new_item);
    new_list_item->AppendTo(prev_list_item);
    next_list_item->AppendTo(new_list_item);
  }
  EmbeddedListItem container_;
};

template<typename ItemField>
class EmbeddedList : public EmbeddedListHead<ItemField> {
 public:
  EmbeddedList() { this->Clear(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_
