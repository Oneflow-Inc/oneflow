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
#ifndef ONEFLOW_CORE_OBJECT_MSG_CHANNEL_H_
#define ONEFLOW_CORE_OBJECT_MSG_CHANNEL_H_

#include <mutex>
#include <condition_variable>
#include "oneflow/core/object_msg/list.h"

namespace oneflow {

namespace intrusive {

enum ChannelStatus {
  kChannelStatusSuccess = 0,
  kChannelStatusErrorClosed,
};

template<typename LinkField>
class Channel {
 public:
  using value_type = typename LinkField::struct_type;

  Channel(const Channel&) = delete;
  Channel(Channel&&) = delete;
  Channel() { this->__Init__(); }
  ~Channel() { this->__Delete__(); }

  void __Init__() {
    list_head_.__Init__();
    is_closed_ = false;
    new (mutex_buff_) std::mutex();
    new (cond_buff_) std::condition_variable();
  }

  bool Empty() {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    return list_head_.empty();
  }

  ChannelStatus EmplaceBack(intrusive::SharedPtr<value_type>&& ptr) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    if (is_closed_) { return kChannelStatusErrorClosed; }
    list_head_.EmplaceBack(std::move(ptr));
    mut_cond()->notify_one();
    return kChannelStatusSuccess;
  }
  ChannelStatus PushBack(value_type* ptr) {
    return EmplaceBack(intrusive::SharedPtr<value_type>(ptr));
  }
  ChannelStatus PopFront(intrusive::SharedPtr<value_type>* ptr) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    mut_cond()->wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kChannelStatusErrorClosed; }
    *ptr = list_head_.PopFront();
    return kChannelStatusSuccess;
  }

  ChannelStatus MoveFrom(intrusive::List<LinkField>* src) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    if (is_closed_) { return kChannelStatusErrorClosed; }
    src->MoveToDstBack(&list_head_);
    mut_cond()->notify_one();
    return kChannelStatusSuccess;
  }

  ChannelStatus MoveTo(intrusive::List<LinkField>* dst) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    mut_cond()->wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kChannelStatusErrorClosed; }
    list_head_.MoveToDstBack(dst);
    return kChannelStatusSuccess;
  }

  ChannelStatus TryMoveTo(intrusive::List<LinkField>* dst) {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    if (list_head_.empty()) { return kChannelStatusSuccess; }
    mut_cond()->wait(lock, [this]() { return (!list_head_.empty()) || is_closed_; });
    if (list_head_.empty()) { return kChannelStatusErrorClosed; }
    list_head_.MoveToDstBack(dst);
    return kChannelStatusSuccess;
  }

  void Close() {
    std::unique_lock<std::mutex> lock(*mut_mutex());
    is_closed_ = true;
    mut_cond()->notify_all();
  }

  void __Delete__() {
    list_head_.Clear();
    using namespace std;
    mut_mutex()->mutex::~mutex();
    mut_cond()->condition_variable::~condition_variable();
  }

 private:
  std::mutex* mut_mutex() { return reinterpret_cast<std::mutex*>(&mutex_buff_[0]); }
  std::condition_variable* mut_cond() {
    return reinterpret_cast<std::condition_variable*>(&cond_buff_[0]);
  }

  intrusive::List<LinkField> list_head_;
  union {
    char mutex_buff_[sizeof(std::mutex)];
    int64_t mutex_buff_align_;
  };
  union {
    char cond_buff_[sizeof(std::condition_variable)];
    int64_t cond_buff_align_;
  };
  bool is_closed_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_CHANNEL_H_
