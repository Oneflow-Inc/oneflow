// The main job test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.

#include "test/test_job.h"

GTEST_API_ int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  //freopen("log.txt", "w", stdout);
  return RUN_ALL_TESTS();
}
