#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/range.h"

namespace oneflow {

namespace test {

namespace {

// clang-format off
BEGIN_OBJECT_MSG(Foo);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int, x);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(link);
END_OBJECT_MSG(Foo);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(FooList);
  // links
  OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(Foo, link, list);
END_OBJECT_MSG(FooList);
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

void CallFromReceiverThread(std::vector<int>* visit, ConditionListFoo* condition_list) {
  ObjectMsgPtr<Foo> foo;
  while (condition_list->PopFront(&foo) == kObjectMsgConditionListStatusSuccess) {
    ++visit->at(foo->x());
  }
}

TEST(ObjectMsgConditionList, 30sender40receiver) {
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
    receivers.push_back(std::thread(CallFromReceiverThread, &visits[i], &condition_list));
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
}  // namespace

}  // namespace test

}  // namespace oneflow
