#include "oneflow/core/common/object_msg_struct.h"
#include "oneflow/core/common/callback.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

TEST(OBJECT_MSG_DEFINE_STRUCT, basic) {
  bool flag = false;
  auto foo = ObjectMsgPtr<CallbackMsg>::New();
  *foo->mut_callback() = [&flag]() { flag = true; };
  foo->callback()();
  ASSERT_TRUE(flag);
}

}  // namespace

}  // namespace test

}  // namespace oneflow
