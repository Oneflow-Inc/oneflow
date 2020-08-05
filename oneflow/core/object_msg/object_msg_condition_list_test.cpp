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
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/range.h"

namespace oneflow {

namespace test {

namespace {

// clang-format off
OBJECT_MSG_BEGIN(Foo);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int, x);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(link);
OBJECT_MSG_END(Foo);
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(FooList);
  // links
  OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(Foo, link, list);
OBJECT_MSG_END(FooList);
// clang-format on

using ConditionListFoo = OBJECT_MSG_CONDITION_LIST(Foo, link);

void CallFromSenderThread(ConditionListFoo* condition_list, Range range) {
  for (int i = range.begin(); i < range.end(); ++i) {
    auto foo = ObjectMsgPtr<Foo>::New();
    foo->set_x(i);
    if (condition_list->EmplaceBack(std::move(foo)) != kObjectMsgConditionListStatusSuccess) {
      break;
    }
  }
}

void CallFromReceiverThreadByPopFront(std::vector<int>* visit, ConditionListFoo* condition_list) {
  ObjectMsgPtr<Foo> foo;
  while (condition_list->PopFront(&foo) == kObjectMsgConditionListStatusSuccess) {
    ++visit->at(foo->x());
  }
}

void CallFromReceiverThreadByMoveTo(std::vector<int>* visit, ConditionListFoo* condition_list) {
  OBJECT_MSG_LIST(Foo, link) tmp_list;
  while (condition_list->MoveTo(&tmp_list) == kObjectMsgConditionListStatusSuccess) {
    OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_list, foo) {
      ++visit->at(foo->x());
      tmp_list.Erase(foo);
    }
  }
}

typedef void (*ThreadHandlerType)(std::vector<int>* visit, ConditionListFoo* condition_list);

void TestConditionList(ThreadHandlerType ThreadHandler) {
  ConditionListFoo condition_list;
  std::vector<std::thread> senders;
  std::vector<std::thread> receivers;
  int sender_num = 30;
  int receiver_num = 40;
  int range_num = 200;
  std::vector<std::vector<int>> visits;
  for (int i = 0; i < receiver_num; ++i) {
    std::vector<int> visit_i;
    for (int j = 0; j < range_num; j++) { visit_i.push_back(0); }
    visits.push_back(visit_i);
  }
  for (int i = 0; i < sender_num; ++i) {
    senders.push_back(std::thread(CallFromSenderThread, &condition_list, Range(0, range_num)));
  }
  for (int i = 0; i < receiver_num; ++i) {
    receivers.push_back(std::thread(ThreadHandler, &visits[i], &condition_list));
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

TEST(ObjectMsgConditionList, 30sender40receiver_pop_front) {
  TestConditionList(&CallFromReceiverThreadByPopFront);
}

TEST(ObjectMsgConditionList, 30sender40receiver_move_to) {
  TestConditionList(&CallFromReceiverThreadByMoveTo);
}

}  // namespace

}  // namespace test

}  // namespace oneflow
