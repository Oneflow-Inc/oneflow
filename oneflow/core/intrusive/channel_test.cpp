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
#include "gtest/gtest.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/intrusive/channel.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/range.h"

namespace oneflow {

namespace intrusive {

namespace test {

namespace {

class Foo final : public intrusive::Base {
 public:
  int x() const { return x_; }
  void set_x(int val) { x_ = val; }

 private:
  Foo() : intrusive_ref_(), x_(), hook_() {}
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }
  intrusive::Ref intrusive_ref_;
  // fields
  int x_;

 public:
  // list hooks
  intrusive::ListHook hook_;
};

using ChannelFoo = intrusive::Channel<INTRUSIVE_FIELD(Foo, hook_)>;

void CallFromSenderThread(ChannelFoo* condition_list, Range range) {
  for (int i = range.begin(); i < range.end(); ++i) {
    auto foo = intrusive::make_shared<Foo>();
    foo->set_x(i);
    if (condition_list->EmplaceBack(std::move(foo)) != intrusive::kChannelStatusSuccess) { break; }
  }
}

void CallFromReceiverThreadByPopFront(std::vector<int>* visit, ChannelFoo* condition_list) {
  intrusive::shared_ptr<Foo> foo;
  while (condition_list->PopFront(&foo) == intrusive::kChannelStatusSuccess) {
    ++visit->at(foo->x());
  }
}

void CallFromReceiverThreadByMoveTo(std::vector<int>* visit, ChannelFoo* condition_list) {
  intrusive::List<INTRUSIVE_FIELD(Foo, hook_)> tmp_list;
  while (condition_list->MoveTo(&tmp_list) == intrusive::kChannelStatusSuccess) {
    INTRUSIVE_FOR_EACH_PTR(foo, &tmp_list) {
      ++visit->at(foo->x());
      tmp_list.Erase(foo);
    }
  }
}

typedef void (*ThreadHandlerType)(std::vector<int>* visit, ChannelFoo* condition_list);

void TestChannel(ThreadHandlerType ThreadHandler) {
  ChannelFoo condition_list;
  std::vector<std::thread> senders;
  std::vector<std::thread> receivers;
  int sender_num = 30;
  int receiver_num = 40;
  int range_num = 200;
  std::vector<std::vector<int>> visits;
  for (int i = 0; i < receiver_num; ++i) {
    std::vector<int> visit_i;
    for (int j = 0; j < range_num; j++) { visit_i.emplace_back(0); }
    visits.emplace_back(visit_i);
  }
  for (int i = 0; i < sender_num; ++i) {
    senders.emplace_back(std::thread(CallFromSenderThread, &condition_list, Range(0, range_num)));
  }
  for (int i = 0; i < receiver_num; ++i) {
    receivers.emplace_back(std::thread(ThreadHandler, &visits[i], &condition_list));
  }
  for (std::thread& this_thread : senders) { this_thread.join(); }
  condition_list.Close();
  for (std::thread& this_thread : receivers) { this_thread.join(); }
  for (int i = 0; i < range_num; ++i) {
    int visit_count = 0;
    for (int j = 0; j < receiver_num; j++) { visit_count += visits[j][i]; }
    ASSERT_EQ(visit_count, sender_num);
  }
}

TEST(Channel, 30sender40receiver_pop_front) { TestChannel(&CallFromReceiverThreadByPopFront); }

TEST(Channel, 30sender40receiver_move_to) { TestChannel(&CallFromReceiverThreadByMoveTo); }

}  // namespace

}  // namespace test

}  // namespace intrusive

}  // namespace oneflow
