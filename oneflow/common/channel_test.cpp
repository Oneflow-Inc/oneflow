#include "common/channel.h"
#include <thread>
#include <vector>
#include "gtest/gtest.h"
#include "common/range.h"

namespace oneflow {

void CallFromSenderThread(Channel<int>* channel, Range range) {
  for (int i = range.begin(); i < range.end(); ++i) {
    if (channel->Send(i) == -1) {
      break;
    }
  }
}

void CallFromReceiverThread(std::vector<int>* visit,
                            Channel<int>* channel) {
  int num = -1;
  int* num_ptr = &num;
  while (channel->Receive(num_ptr) == 0) {
    ++visit->at(*num_ptr);
  }
}


TEST(Channel, 30sender40receiver) {
  Channel<int> channel;
  std::vector<std::thread> senders;
  std::vector<std::thread> receivers;
  int sender_num = 30;
  int receiver_num = 40;
  int range_num = 200;
  std::vector<std::vector<int>> visits;
  for (int i = 0; i < receiver_num; ++i) {
    std::vector<int> visit_i;
    for (int j = 0; j < range_num; j++) {
      visit_i.push_back(0);
    }
    visits.push_back(visit_i);
  }
  for (int i = 0; i < sender_num; ++i) {
    senders.push_back(std::thread(CallFromSenderThread,
                                  &channel,
                                  Range(0, range_num)));
  }
  for (int i = 0; i < receiver_num; ++i) {
    receivers.push_back(std::thread(CallFromReceiverThread,
                                    &visits[i],
                                    &channel));
  }
  for (std::thread& this_thread : senders) {
    this_thread.join();
  }
  channel.CloseSendEnd();
  for (std::thread& this_thread : receivers) {
    this_thread.join();
  }
  channel.CloseReceiveEnd();
  for (int i = 0; i < range_num; ++i) {
    int visit_count = 0;
    for (int j = 0; j < receiver_num; j++) {
      visit_count += visits[j][i];
    }
    ASSERT_EQ(visit_count, sender_num);
  }
}

}  // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
