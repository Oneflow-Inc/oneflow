#include "oneflow/core/common/object_msg_struct.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

// clang-format off
OBJECT_MSG_BEGIN(WrapperFoo);
  OBJECT_MSG_DEFINE_STRUCT(std::function<void()>, function);
OBJECT_MSG_END(WrapperFoo);
// clang-format on

TEST(OBJECT_MSG_DEFINE_STRUCT, basic) {
  bool flag = false;
  auto foo = ObjectMsgPtr<WrapperFoo>::New();
  *foo->mut_function() = [&flag]() { flag = true; };
  foo->function()();
  ASSERT_TRUE(flag);
}

}  // namespace

}  // namespace test

}  // namespace oneflow
