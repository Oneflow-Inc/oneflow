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
#ifndef ONEFLOW_CORE_INTRUSIVE_CHANNEL_H_
#define ONEFLOW_CORE_INTRUSIVE_CHANNEL_H_

#include <mutex>
#include <condition_variable>
#include "oneflow/core/intrusive/list.h"

namespace oneflow {

namespace intrusive {

enum ChannelStatus {
  kChannelStatusSuccess = 0,
  kChannelStatusErrorClosed,
};

template<typename HookField>
class Channel {
 public:
  using value_type = typename HookField::struct_type;

  Channel(const Channel&) = delete;
  Channel(Channel&&) = delete;
  Channel() : list_head_(), mutex_(), cond_(), is_closed_(false) {}
  ~Channel() = default;

  bool Empty() {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    return list_head_.empty();
  }

  ChannelStatus EmplaceBack(intrusive::shared_ptr<value_type>&& ptr) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    if (is_closed_) { return kChannelStatusErrorClosed; }
    list_head_.EmplaceBack(std::move(ptr));
    mut_cond()->notify_one();
    return kChannelStatusSuccess;
  }
  ChannelStatus PushBack(value_type* ptr) {
    return EmplaceBack(intrusive::shared_ptr<value_type>(ptr));
  }
  ChannelStatus PopFront(intrusive::shared_ptr<value_type>* ptr) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    mut_cond()->wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kChannelStatusErrorClosed; }
    *ptr = list_head_.PopFront();
    return kChannelStatusSuccess;
  }

  ChannelStatus MoveFrom(List<HookField>* src) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    if (is_closed_) { return kChannelStatusErrorClosed; }
    src->MoveToDstBack(&list_head_);
    mut_cond()->notify_one();
    return kChannelStatusSuccess;
  }

  ChannelStatus MoveTo(List<HookField>* dst) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    mut_cond()->wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kChannelStatusErrorClosed; }
    list_head_.MoveToDstBack(dst);
    return kChannelStatusSuccess;
  }

  ChannelStatus TryMoveTo(List<HookField>* dst) {
    if (list_head_.size() == 0) { return kChannelStatusSuccess; }
    std::unique_lock<std::mutex> lock(*mut_mutex());
    list_head_.MoveToDstBack(dst);
    return kChannelStatusSuccess;
  }

  void Close() {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    is_closed_ = true;
    mut_cond()->notify_all();
  }

 private:
  std::mutex* mut_mutex() { return &mutex_; }
  std::condition_variable* mut_cond() { return &cond_; }

  List<HookField> list_head_;
  std::mutex mutex_;
  std::condition_variable cond_;
  bool is_closed_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_CHANNEL_H_
