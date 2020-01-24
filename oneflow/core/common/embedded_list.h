#ifndef ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_
#define ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_

#include "oneflow/core/common/struct_traits.h"
#include <glog/logging.h>

namespace oneflow {

struct PodEmbeddedListItem {
 public:
  PodEmbeddedListItem* prev() const { return prev_; }
  PodEmbeddedListItem* next() const { return next_; }

  void AppendTo(PodEmbeddedListItem* prev) {
    prev->set_next(this);
    this->set_prev(prev);
  }
  void Clear() {
    prev_ = this;
    next_ = this;
  }

 private:
  void set_prev(PodEmbeddedListItem* prev) { prev_ = prev; }
  void set_next(PodEmbeddedListItem* next) { next_ = next; }

  PodEmbeddedListItem* prev_;
  PodEmbeddedListItem* next_;
};

class EmbeddedListItem final : public PodEmbeddedListItem {
 public:
  EmbeddedListItem() { this->Clear(); }
};

class PodEmbeddedListHead {
 public:
  void Clear() { container_.Clear(); }

 protected:
  const PodEmbeddedListItem& container() const { return container_; }
  PodEmbeddedListItem* mut_container() { return &container_; }

 private:
  PodEmbeddedListItem container_;
};

template<typename ItemField>
class PodEmbeddedListHeadIf : public PodEmbeddedListHead {
 public:
  using item_type = typename ItemField::struct_type;
  static_assert(std::is_same<typename ItemField::field_type, PodEmbeddedListItem>::value,
                "no PodEmbeddedListItem found in item");

  void Erase(item_type* item) {
    CHECK_NE(item, end_item());
    PodEmbeddedListItem* list_item = ItemField::FieldPtr4StructPtr(item);
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

  bool empty() const { return &begin_item() == &end_item(); }
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

 private:
  void InsertAfter(item_type* prev_item, item_type* new_item) {
    PodEmbeddedListItem* prev_list_item = ItemField::FieldPtr4StructPtr(prev_item);
    PodEmbeddedListItem* next_list_item = prev_list_item->next();
    PodEmbeddedListItem* new_list_item = ItemField::FieldPtr4StructPtr(new_item);
    new_list_item->AppendTo(prev_list_item);
    next_list_item->AppendTo(new_list_item);
  }
};

template<typename ItemField>
class EmbeddedList : public PodEmbeddedListHeadIf<ItemField> {
 public:
  EmbeddedList() { this->Clear(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_
