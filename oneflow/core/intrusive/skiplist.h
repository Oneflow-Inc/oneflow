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
#ifndef ONEFLOW_CORE_INTRUSIVE_INTRUSIVE_SKIPLIST_H_
#define ONEFLOW_CORE_INTRUSIVE_INTRUSIVE_SKIPLIST_H_

#include "oneflow/core/intrusive/intrusive_core.h"
#include "oneflow/core/intrusive/skiplist_entry.h"

namespace oneflow {

#define INTRUSIVE_SKIPLIST_FOR_EACH(skiplist_ptr, elem)                                         \
  _INTRUSIVE_SKIPLIST_FOR_EACH(std::remove_pointer<decltype(skiplist_ptr)>::type, skiplist_ptr, \
                               elem)

#define INTRUSIVE_SKIPLIST_FOR_EACH_PTR(skiplist_ptr, elem)                           \
  _INTRUSIVE_SKIPLIST_FOR_EACH_PTR(std::remove_pointer<decltype(skiplist_ptr)>::type, \
                                   skiplist_ptr, elem)

#define INTRUSIVE_SKIPLIST_UNSAFE_FOR_EACH_PTR(skiplist_ptr, elem)                           \
  _INTRUSIVE_SKIPLIST_UNSAFE_FOR_EACH_PTR(std::remove_pointer<decltype(skiplist_ptr)>::type, \
                                          skiplist_ptr, elem)
// details

#define _INTRUSIVE_SKIPLIST_FOR_EACH(skiplist_type, skiplist_ptr, elem)                      \
  for (intrusive::SharedPtr<skiplist_type::value_type> elem, *end_if_not_null = nullptr;     \
       end_if_not_null == nullptr; end_if_not_null = nullptr, ++end_if_not_null)             \
  LIST_ENTRY_FOR_EACH_WITH_EXPR(                                                             \
      (StructField<                                                                          \
          skiplist_type, intrusive::ListEntry,                                               \
          skiplist_type::ContainerLevelZeroLinkOffset()>::FieldPtr4StructPtr(skiplist_ptr)), \
      skiplist_type::elem_level0_entry_struct_field, elem_ptr, (elem.Reset(elem_ptr), true))

#define _INTRUSIVE_SKIPLIST_FOR_EACH_PTR(skiplist_type, skiplist_ptr, elem)                  \
  LIST_ENTRY_FOR_EACH(                                                                       \
      (StructField<                                                                          \
          skiplist_type, intrusive::ListEntry,                                               \
          skiplist_type::ContainerLevelZeroLinkOffset()>::FieldPtr4StructPtr(skiplist_ptr)), \
      skiplist_type::elem_level0_entry_struct_field, elem)

#define _INTRUSIVE_SKIPLIST_UNSAFE_FOR_EACH_PTR(skiplist_type, skiplist_ptr, elem)           \
  LIST_ENTRY_UNSAFE_FOR_EACH(                                                                \
      (StructField<                                                                          \
          skiplist_type, intrusive::ListEntry,                                               \
          skiplist_type::ContainerLevelZeroLinkOffset()>::FieldPtr4StructPtr(skiplist_ptr)), \
      skiplist_type::elem_level0_entry_struct_field, elem)

namespace intrusive {

template<typename ElemKeyField>
class SkipList {
 public:
  SkipList(const SkipList&) = delete;
  SkipList(SkipList&&) = delete;

  SkipList() { this->__Init__(); }
  ~SkipList() { this->Clear(); }

  using value_type = typename ElemKeyField::struct_type;
  using key_type = typename ElemKeyField::field_type::key_type;
  using elem_key_level0_entry_struct_field =
      StructField<typename ElemKeyField::field_type, intrusive::ListEntry,
                  ElemKeyField::field_type::LevelZeroLinkOffset()>;
  using elem_level0_entry_struct_field =
      typename ComposeStructField<ElemKeyField, elem_key_level0_entry_struct_field>::type;
  template<typename Enabled = void>
  static constexpr int ContainerLevelZeroLinkOffset() {
    return offsetof(SkipList, skiplist_head_)
           + intrusive::SkipListHead<ElemKeyField>::ContainerLevelZeroLinkOffset();
  }

  void __Init__() { skiplist_head_.__Init__(); }

  std::size_t size() const { return skiplist_head_.size(); }
  bool empty() const { return skiplist_head_.empty(); }
  value_type* Begin() { return skiplist_head_.Begin(); }
  intrusive::SharedPtr<value_type> Find(const key_type& key) {
    intrusive::SharedPtr<value_type> ret;
    ret.Reset(skiplist_head_.Find(key));
    return ret;
  }
  value_type* FindPtr(const key_type& key) { return skiplist_head_.Find(key); }
  const value_type* FindPtr(const key_type& key) const { return skiplist_head_.Find(key); }
  bool EqualsEnd(const intrusive::SharedPtr<value_type>& ptr) { return !ptr; }
  void Erase(const key_type& key) { PtrUtil::ReleaseRef(skiplist_head_.Erase(key)); }
  void Erase(value_type* elem_ptr) {
    skiplist_head_.Erase(elem_ptr);
    PtrUtil::ReleaseRef(elem_ptr);
  }
  std::pair<intrusive::SharedPtr<value_type>, bool> Insert(value_type* elem_ptr) {
    value_type* ret_elem = nullptr;
    bool success = false;
    std::tie(ret_elem, success) = skiplist_head_.Insert(elem_ptr);
    std::pair<intrusive::SharedPtr<value_type>, bool> ret;
    ret.first.Reset(ret_elem);
    ret.second = success;
    if (success) { PtrUtil::Ref(elem_ptr); }
    return ret;
  }

  void Clear() {
    skiplist_head_.Clear([](value_type* elem) { PtrUtil::ReleaseRef(elem); });
  }

 private:
  intrusive::SkipListHead<ElemKeyField> skiplist_head_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_INTRUSIVE_SKIPLIST_H_
