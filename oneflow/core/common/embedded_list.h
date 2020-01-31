#ifndef ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_
#define ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_

#include "oneflow/core/common/struct_traits.h"
#include <glog/logging.h>

namespace oneflow {

struct EmbeddedListItem {
 public:
  EmbeddedListItem* prev() const { return prev_; }
  EmbeddedListItem* next() const { return next_; }

  void AppendTo(EmbeddedListItem* prev) {
    prev->set_next(this);
    this->set_prev(prev);
  }
  void __Init__() {
    prev_ = this;
    next_ = this;
  }
  bool empty() const { return prev_ == this || next_ == this; }

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

  std::size_t size() const { return size_; }
  bool empty() const {
    bool list_empty = (&begin_item() == &end_item());
    bool size_empty = (size_ == 0);
    CHECK_EQ(list_empty, size_empty);
    return size_empty;
  }
  const item_type& begin_item() const { return next_item(end_item()); }
  const item_type& last_item() const { return prev_item(end_item()); }
  const item_type& end_item() const { return *ItemField::StructPtr4FieldPtr(&container()); }
  const item_type& next_item(const item_type& current) const {
    return *ItemField::StructPtr4FieldPtr(ItemField::FieldPtr4StructPtr(&current)->next());
  }
  const item_type& prev_item(const item_type& current) const {
    return *ItemField::StructPtr4FieldPtr(ItemField::FieldPtr4StructPtr(&current)->prev());
  }

  item_type* begin_item() { return next_item(end_item()); }
  item_type* last_item() { return prev_item(end_item()); }
  item_type* end_item() { return ItemField::StructPtr4FieldPtr(mut_container()); }
  item_type* next_item(item_type* current) {
    return ItemField::StructPtr4FieldPtr(ItemField::FieldPtr4StructPtr(current)->next());
  }
  item_type* prev_item(item_type* current) {
    return ItemField::StructPtr4FieldPtr(ItemField::FieldPtr4StructPtr(current)->prev());
  }
  void __Init__() {
    container_.__Init__();
    size_ = 0;
  }

  void Erase(item_type* item) {
    CHECK_GT(size_, 0);
    CHECK_NE(item, end_item());
    EmbeddedListItem* list_item = ItemField::FieldPtr4StructPtr(item);
    list_item->next()->AppendTo(list_item->prev());
    list_item->__Init__();
    --size_;
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

 private:
  void InsertAfter(item_type* prev_item, item_type* new_item) {
    EmbeddedListItem* prev_list_item = ItemField::FieldPtr4StructPtr(prev_item);
    EmbeddedListItem* next_list_item = prev_list_item->next();
    EmbeddedListItem* new_list_item = ItemField::FieldPtr4StructPtr(new_item);
    new_list_item->AppendTo(prev_list_item);
    next_list_item->AppendTo(new_list_item);
    ++size_;
  }
  const EmbeddedListItem& container() const { return container_; }
  EmbeddedListItem* mut_container() { return &container_; }

 private:
  EmbeddedListItem container_;
  std::size_t size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_
