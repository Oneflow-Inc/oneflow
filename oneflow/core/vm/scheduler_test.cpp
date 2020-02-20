#include <iostream>
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/common/object_msg_reflection.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

TEST(VmScheduler, ToDot) {
  std::string dot_str = ObjectMsgListReflection<VmScheduler>().ToDot("VmScheduler");
  //  std::cout << std::endl;
  //  std::cout << dot_str << std::endl;
  //  std::cout << std::endl;
}

}  // namespace

}  // namespace test

}  // namespace oneflow
