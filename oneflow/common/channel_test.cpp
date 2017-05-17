#include <common/channel.h>
#include <gtest/gtest.h>

namespace oneflow {



TEST(Channel, 2sender2reciver) {
  Channel<int> channel;

}

}  // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
