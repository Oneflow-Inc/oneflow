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

#include "oneflow/core/intrusive/ref.h"
#include "oneflow/core/intrusive/skiplist_hook.h"

namespace oneflow {

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
  using elem_key_level0_hook_struct_field =
      OffsetStructField<typename ElemKeyField::field_type, intrusive::ListHook,
                        ElemKeyField::field_type::LevelZeroHookOffset()>;
  using iterator_struct_field = ComposeStructField<ElemKeyField, elem_key_level0_hook_struct_field>;
  template<typename Enabled = void>
  static constexpr int IteratorHookOffset() {
    return offsetof(SkipList, skiplist_head_)
           + intrusive::SkipListHead<ElemKeyField>::IteratorHookOffset();
  }

  void __Init__() { skiplist_head_.__Init__(); }

  std::size_t size() const { return skiplist_head_.size(); }
  bool empty() const { return skiplist_head_.empty(); }
  value_type* Begin() { return skiplist_head_.Begin(); }
  intrusive::shared_ptr<value_type> Find(const key_type& key) {
    intrusive::shared_ptr<value_type> ret;
    ret.Reset(skiplist_head_.Find(key));
    return ret;
  }
  value_type* FindPtr(const key_type& key) { return skiplist_head_.Find(key); }
  const value_type* FindPtr(const key_type& key) const { return skiplist_head_.Find(key); }
  bool EqualsEnd(const intrusive::shared_ptr<value_type>& ptr) { return !ptr; }
  void Erase(const key_type& key) { Ref::DecreaseRef(skiplist_head_.Erase(key)); }
  void Erase(value_type* elem_ptr) {
    skiplist_head_.Erase(elem_ptr);
    Ref::DecreaseRef(elem_ptr);
  }
  std::pair<intrusive::shared_ptr<value_type>, bool> Insert(value_type* elem_ptr) {
    value_type* ret_elem = nullptr;
    bool success = false;
    std::tie(ret_elem, success) = skiplist_head_.Insert(elem_ptr);
    std::pair<intrusive::shared_ptr<value_type>, bool> ret;
    ret.first.Reset(ret_elem);
    ret.second = success;
    if (success) { Ref::IncreaseRef(elem_ptr); }
    return ret;
  }

  void Clear() {
    skiplist_head_.Clear([](value_type* elem) { Ref::DecreaseRef(elem); });
  }

 private:
  intrusive::SkipListHead<ElemKeyField> skiplist_head_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_INTRUSIVE_SKIPLIST_H_
