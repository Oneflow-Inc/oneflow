#include "oneflow/core/common/object_msg_volatile.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgVolatileFoo);
  OBJECT_MSG_DEFINE_VOLATILE(int, bar);
END_OBJECT_MSG(ObjectMsgVolatileFoo);
// clang-format on

TEST(object_msg_volatile, simple) {
  auto obj = OBJECT_MSG_PTR(ObjectMsgVolatileFoo)::New();
  obj->set_bar(9527);
  ASSERT_EQ(obj->bar(), 9527);
}

}  // namespace test

}  // namespace oneflow
