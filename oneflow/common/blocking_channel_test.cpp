#include "common/blocking_channel.h"
#include "gtest/gtest.h"

namespace oneflow {

void write() {

}

void read() {}

TEST(BlockingChannel, blocking_channel_2_writer_3_reader_test) {
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
