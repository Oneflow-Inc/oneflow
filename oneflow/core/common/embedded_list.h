#ifndef ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_
#define ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_

#include "oneflow/core/common/struct_traits.h"
#include <glog/logging.h>

namespace oneflow {

struct EmbeddedListLink {
 public:
  EmbeddedListLink* prev() const { return prev_; }
  EmbeddedListLink* next() const { return next_; }

  void __Init__() { Clear(); }
  void Clear() {
    prev_ = this;
    next_ = this;
  }

  bool empty() const { return prev_ == this || next_ == this; }
  void AppendTo(EmbeddedListLink* prev) {
    prev->set_next(this);
    this->set_prev(prev);
  }
  void InsertAfter(EmbeddedListLink* prev) {
    auto* next = prev->next();
    this->AppendTo(prev);
    next->AppendTo(this);
  }
  void Erase() {
    next_->AppendTo(prev_);
    Clear();
  }

  bool nullptr_empty() const { return prev_ == nullptr && next_ == nullptr; }

  void NullptrClear() {
    prev_ = nullptr;
    next_ = nullptr;
  }

 private:
  void set_prev(EmbeddedListLink* prev) { prev_ = prev; }
  void set_next(EmbeddedListLink* next) { next_ = next; }

  EmbeddedListLink* prev_;
  EmbeddedListLink* next_;
};

template<typename LinkField>
class EmbeddedListHead {
 public:
  using value_type = typename LinkField::struct_type;
  static_assert(std::is_same<typename LinkField::field_type, EmbeddedListLink>::value,
                "no EmbeddedListLink found");

  std::size_t size() const { return size_; }
  bool empty() const {
    bool list_empty = (&Begin() == &End());
    bool size_empty = (size_ == 0);
    CHECK_EQ(list_empty, size_empty);
    return size_empty;
  }
  const value_type& Begin() const { return Next(End()); }
  const value_type& ReverseBegin() const { return Prev(End()); }
  const value_type& End() const { return *LinkField::StructPtr4FieldPtr(&container()); }
  const value_type& Next(const value_type& current) const {
    return *LinkField::StructPtr4FieldPtr(LinkField::FieldPtr4StructPtr(&current)->next());
  }
  const value_type& Prev(const value_type& current) const {
    return *LinkField::StructPtr4FieldPtr(LinkField::FieldPtr4StructPtr(&current)->prev());
  }

  value_type* Begin() { return Next(End()); }
  value_type* Last() { return Prev(End()); }
  value_type* End() { return LinkField::StructPtr4FieldPtr(mut_container()); }
  value_type* Next(value_type* current) {
    return LinkField::StructPtr4FieldPtr(LinkField::FieldPtr4StructPtr(current)->next());
  }
  value_type* Prev(value_type* current) {
    return LinkField::StructPtr4FieldPtr(LinkField::FieldPtr4StructPtr(current)->prev());
  }
  void __Init__() { Clear(); }

  void Clear() {
    container_.__Init__();
    size_ = 0;
  }

  void Erase(value_type* elem) {
    CHECK_GT(size_, 0);
    CHECK_NE(elem, End());
    EmbeddedListLink* list_link = LinkField::FieldPtr4StructPtr(elem);
    list_link->Erase();
    --size_;
  }
  void MoveToDstBack(value_type* elem, EmbeddedListHead* dst) {
    CHECK(!container_.empty());
    auto* dst_rbegin = dst->container_.prev();
    auto* dst_end = &dst->container_;
    EmbeddedListLink* elem_link = LinkField::FieldPtr4StructPtr(elem);
    elem_link->next()->AppendTo(elem_link->prev());
    elem_link->AppendTo(dst_rbegin);
    dst_end->AppendTo(elem_link);
    --size_;
    ++dst->size_;
  }
  void MoveToDstFront(value_type* elem, EmbeddedListHead* dst) {
    CHECK(!container_.empty());
    auto* dst_end = &dst->container_;
    auto* dst_begin = dst->container_.next();
    EmbeddedListLink* elem_link = LinkField::FieldPtr4StructPtr(elem);
    elem_link->next()->AppendTo(elem_link->prev());
    elem_link->AppendTo(dst_end);
    dst_begin->AppendTo(elem_link);
    --size_;
    ++dst->size_;
  }
  void PushBack(value_type* elem) { InsertAfter(Last(), elem); }
  void PushFront(value_type* elem) { InsertAfter(End(), elem); }
  value_type* PopBack() {
    CHECK(!empty());
    value_type* last = Last();
    Erase(last);
    return last;
  }
  value_type* PopFront() {
    CHECK(!empty());
    value_type* first = Begin();
    Erase(first);
    return first;
  }
  void MoveToDstBack(EmbeddedListHead* dst) {
    if (container_.empty()) { return; }
    auto* dst_last = dst->container_.prev();
    auto* dst_end = &dst->container_;
    auto* this_first = container_.next();
    auto* this_last = container_.prev();
    this_first->AppendTo(dst_last);
    dst_end->AppendTo(this_last);
    dst->size_ += size();
    this->Clear();
  }

 private:
  void InsertAfter(value_type* prev_elem, value_type* new_elem) {
    EmbeddedListLink* prev_list_link = LinkField::FieldPtr4StructPtr(prev_elem);
    EmbeddedListLink* next_list_link = prev_list_link->next();
    EmbeddedListLink* new_list_link = LinkField::FieldPtr4StructPtr(new_elem);
    new_list_link->AppendTo(prev_list_link);
    next_list_link->AppendTo(new_list_link);
    ++size_;
  }
  const EmbeddedListLink& container() const { return container_; }
  EmbeddedListLink* mut_container() { return &container_; }

 private:
  EmbeddedListLink container_;
  std::size_t size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EMBEDDED_LIST_H_
