#ifndef ONEFLOW_CORE_COMMON_EMBEDDED_SKIPLIST_H_
#define ONEFLOW_CORE_COMMON_EMBEDDED_SKIPLIST_H_

#include <array>
#include <random>
#include <glog/logging.h>
#include "oneflow/core/common/struct_traits.h"

namespace oneflow {

struct EmbeddedSingleListIterator final {
 public:
  void __Init__() { next_ = nullptr; }

  bool empty() const { return next_ == nullptr; }

  EmbeddedSingleListIterator* next() const { return next_; }

  void InsertNextIterator(EmbeddedSingleListIterator* next) {
    CHECK(next->empty());
    next->next_ = next_;
    next_ = next;
  }

  void EraseNextIterator() {
    if (next_ == nullptr) { return; }
    EmbeddedSingleListIterator* next = next_;
    next_ = next_->next_;
    next->next_ = nullptr;
  }

 private:
  EmbeddedSingleListIterator* next_;
};

template<int max_level>
struct EmbeddedSkipListIterator final {
 public:
  using self_type = EmbeddedSkipListIterator<max_level>;
  void __Init__() {
    for (auto& link : links_) { link.__Init__(); }
  }
  void InsertAfter(const std::array<EmbeddedSingleListIterator*, max_level>& prev_links,
                   int levels) {
    for (int i = 0; i < levels; ++i) { prev_links[i]->InsertNextIterator(&links_[i]); }
  }
  void GetSelfLinksPointers(std::array<EmbeddedSingleListIterator*, max_level>* links_pointers) {
    for (int i = 0; i < max_level; ++i) { (*links_pointers)[i] = &links_[i]; }
  }
  static EmbeddedSkipListIterator* ThisPtr4SingleListPtr(EmbeddedSingleListIterator* slist_ptr,
                                                         int level) {
    auto* links_ptr = (std::array<EmbeddedSingleListIterator, max_level>*)(slist_ptr - level);
    return StructField<self_type, decltype(links_), LinksOffset()>::StructPtr4FieldPtr(links_ptr);
  }

 private:
  template<typename Enabled = void>
  static constexpr int LinksOffset() {
    return offsetof(self_type, links_);
  }

  std::array<EmbeddedSingleListIterator, max_level> links_;
};

template<typename SkipListKey>
class EmbeddedSkipListVisitor final {
 public:
  using key_type = typename SkipListKey::key_type;
  static const int max_level = SkipListKey::max_level;

  EmbeddedSkipListVisitor(const key_type* key, EmbeddedSkipListIterator<max_level>* head)
      : key_(key) {
    head->GetSelfLinksPointers(&links_pointers_);
  }
  const key_type& key() const { return *key_; }

  SkipListKey* Next() { return SkipListKey::ThisPtr4SingleListPtr(links_pointers_[0]->next(), 0); }

  void SearchLastIteratorLessThanMe() {
    for (int level = max_level - 1; level >= 0; --level) {
      SearchLastIteratorLessThanMe(&links_pointers_[level], level);
      if (level > 0) { links_pointers_[level - 1] = links_pointers_[level] - 1; }
    }
  }

  void InsertNextIterator(SkipListKey* elem_key) {
    CHECK(!(elem_key->key() < key()));
    std::mt19937 rand{std::random_device{}()};
    int num_level = 1;
    for (; (rand() % 2 == 0) && num_level <= max_level; ++num_level)
      ;
    elem_key->mut_skiplist_iter()->InsertAfter(links_pointers_, num_level);
  }

  void EraseNextIterator() {
    EmbeddedSingleListIterator* bottom_slist_iter = links_pointers_[0]->next();
    for (int level = 0; level < max_level; ++level) {
      if (links_pointers_[level]->next() == bottom_slist_iter + level) {
        links_pointers_[level]->EraseNextIterator();
      } else {
        CHECK(bottom_slist_iter[level].next() == nullptr);
      }
    }
  }

 private:
  void SearchLastIteratorLessThanMe(EmbeddedSingleListIterator** link_ptr, int level) {
    for (;; *link_ptr = (*link_ptr)->next()) {
      if ((*link_ptr)->next() == nullptr) { return; }
      auto* elem_key = SkipListKey::ThisPtr4SingleListPtr((*link_ptr)->next(), level);
      if (!(elem_key->key() < this->key())) { return; }
    }
  }

  const key_type* key_;
  std::array<EmbeddedSingleListIterator*, max_level> links_pointers_;
};

template<typename T, int N = 20>
struct EmbeddedSkipListKey {
 public:
  using self_type = EmbeddedSkipListKey<T, N>;
  using iter_type = EmbeddedSkipListIterator<N>;
  using key_type = T;
  static const int max_level = N;

  void __Init__() { skiplist_iter_.__Init__(); }

  const T& key() const { return key_; }
  T* mut_key() { return &key_; }

  const EmbeddedSkipListIterator<N>& skiplist_iter() const { return skiplist_iter_; }
  EmbeddedSkipListIterator<N>* mut_skiplist_iter() { return &skiplist_iter_; }

  static EmbeddedSkipListKey* ThisPtr4SingleListPtr(EmbeddedSingleListIterator* slist_ptr,
                                                    int level) {
    using FieldUtil = StructField<self_type, EmbeddedSkipListIterator<N>, SkipListIteratorOffset()>;
    auto* skip_list_ptr = EmbeddedSkipListIterator<N>::ThisPtr4SingleListPtr(slist_ptr, level);
    return FieldUtil::StructPtr4FieldPtr(skip_list_ptr);
  }

 private:
  template<typename Enabled = void>
  static constexpr int SkipListIteratorOffset() {
    return offsetof(self_type, skiplist_iter_);
  }

  EmbeddedSkipListIterator<N> skiplist_iter_;
  T key_;
};

template<typename ElemKeyField, int N = 20>
class EmbeddedSkipListHead {
 public:
  using elem_type = typename ElemKeyField::struct_type;
  using elem_key_type = typename ElemKeyField::field_type;
  using key_type = typename elem_key_type::key_type;
  static const int max_level = N;

  void __Init__() {
    skiplist_head_.__Init__();
    size_ = 0;
  }

  std::size_t size() const { return size_; }
  bool empty() { return size_ == 0; }

  elem_type* Find(const key_type& key) {
    EmbeddedSkipListVisitor<elem_key_type> visitor(&key, &skiplist_head_);
    visitor.SearchLastIteratorLessThanMe();
    elem_key_type* elem_key = visitor.Next();
    if (elem_key == nullptr || !(elem_key->key() == key)) { return nullptr; }
    return ElemKeyField::StructPtr4FieldPtr(elem_key);
  }
  void Erase(const key_type& key) {
    EmbeddedSkipListVisitor<elem_key_type> visitor(&key, &skiplist_head_);
    visitor.SearchLastIteratorLessThanMe();
    elem_key_type* searched = visitor.Next();
    CHECK_NOTNULL(searched);
    CHECK(searched->key() == key);
    visitor.EraseNextIterator();
    --size_;
  }
  void Erase(elem_type* elem) {
    elem_key_type* elem_iter = ElemKeyField::FieldPtr4StructPtr(elem);
    EmbeddedSkipListVisitor<elem_key_type> visitor(&elem_iter->key(), &skiplist_head_);
    visitor.SearchLastIteratorLessThanMe();
    elem_key_type* searched = visitor.Next();
    CHECK_NOTNULL(searched);
    CHECK(searched == elem_iter);
    visitor.EraseNextIterator();
    --size_;
  }
  // return true if success
  bool Insert(elem_type* elem) {
    elem_key_type* new_elem_key = ElemKeyField::FieldPtr4StructPtr(elem);
    EmbeddedSkipListVisitor<elem_key_type> visitor(&new_elem_key->key(), &skiplist_head_);
    visitor.SearchLastIteratorLessThanMe();
    elem_key_type* searched = visitor.Next();
    visitor.InsertNextIterator(new_elem_key);
    bool ret = ((searched == nullptr) || !(searched->key() == new_elem_key->key()));
    if (ret) { ++size_; }
    return ret;
  }

 private:
  EmbeddedSkipListIterator<N> skiplist_head_;
  std::size_t size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EMBEDDED_SKIPLIST_H_
