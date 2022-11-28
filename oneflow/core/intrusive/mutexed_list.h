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
#ifndef ONEFLOW_CORE_INTRUSIVE_MUTEXED_LIST_H_
#define ONEFLOW_CORE_INTRUSIVE_MUTEXED_LIST_H_

#include <mutex>
#include "oneflow/core/intrusive/list.h"

namespace oneflow {

namespace intrusive {

template<typename HookField>
class MutexedList {
 public:
  using value_type = typename HookField::struct_type;
  using list_type = List<HookField>;

  MutexedList(const MutexedList&) = delete;
  MutexedList(MutexedList&&) = delete;
  explicit MutexedList(std::mutex* mutex) { this->__Init__(mutex); }
  ~MutexedList() { this->Clear(); }

  std::size_t thread_unsafe_size() const { return list_head_.size(); }
  std::size_t size() const {
    std::unique_lock<std::mutex> lock(*mutex_);
    return list_head_.size();
  }
  bool empty() const {
    std::unique_lock<std::mutex> lock(*mutex_);
    return list_head_.empty();
  }

  void __Init__(std::mutex* mutex) {
    list_head_.__Init__();
    mutex_ = mutex;
  }

  void EmplaceBack(intrusive::shared_ptr<value_type>&& ptr) {
    std::unique_lock<std::mutex> lock(*mutex_);
    return list_head_.EmplaceBack(std::move(ptr));
  }
  void EmplaceFront(intrusive::shared_ptr<value_type>&& ptr) {
    std::unique_lock<std::mutex> lock(*mutex_);
    return list_head_.EmplaceFront(std::move(ptr));
  }
  void PushBack(value_type* ptr) { EmplaceBack(intrusive::shared_ptr<value_type>(ptr)); }
  void PushFront(value_type* ptr) { EmplaceFront(intrusive::shared_ptr<value_type>(ptr)); }
  intrusive::shared_ptr<value_type> PopBack() {
    std::unique_lock<std::mutex> lock(*mutex_);
    return list_head_.PopBack();
  }
  intrusive::shared_ptr<value_type> PopFront() {
    std::unique_lock<std::mutex> lock(*mutex_);
    return list_head_.PopFront();
  }

  // Returns true if old list is empty.
  bool MoveFrom(list_type* src) {
    std::unique_lock<std::mutex> lock(*mutex_);
    return ThreadUnsafeMoveFrom(src);
  }

  // Returns true if old list is empty.
  bool ThreadUnsafeMoveFrom(list_type* src) {
    bool old_list_empty = list_head_.empty();
    src->MoveToDstBack(&list_head_);
    return old_list_empty;
  }

  void MoveTo(list_type* dst) {
    std::unique_lock<std::mutex> lock(*mutex_);
    list_head_.MoveToDstBack(dst);
  }

  void ThreadUnsafeMoveTo(list_type* dst) { list_head_.MoveToDstBack(dst); }

  void Clear() {
    std::unique_lock<std::mutex> lock(*mutex_);
    list_head_.Clear();
  }

 private:
  list_type list_head_;
  std::mutex* mutex_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_MUTEXED_LIST_H_
