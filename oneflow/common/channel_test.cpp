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


TEST(Channel, 3sender4receiver) {
  Channel<int> channel;
  std::vector<int> visit;
  std::vector<std::thread> senders;
  std::vector<std::thread> receivers;
  int sender_num = 3;
  int receiver_num = 4;
  for (int i = 0; i < 5; ++i) {
    visit.push_back(0);
  }
  for (int i = 0; i < sender_num; ++i) {
    senders.push_back(std::thread(CallFromSenderThread,
                            &channel,
                            Range(0, 5)));
  }
  for (int i = 0; i < receiver_num; ++i) {
    receivers.push_back(std::thread(CallFromReceiverThread,
                                    &visit,
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
  for (int visit_count : visit) {
    ASSERT_EQ(visit_count, sender_num);
  }
}

}  // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
