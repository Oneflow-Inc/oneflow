#include "oneflow/core/common/object_msg_raw_ptr.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

typedef bool (*Ok)();

bool TestOk() { return true; }

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgFunctionPtrFoo);
  OBJECT_MSG_DEFINE_RAW_PTR(Ok, ok);
END_OBJECT_MSG(ObjectMsgFunctionPtrFoo);
// clang-format on

TEST(object_msg_raw_ptr, function_pointer) {
  auto obj = ObjectMsgPtr<ObjectMsgFunctionPtrFoo>::New();
  obj->set_ok(&TestOk);
  ASSERT_TRUE((*obj->ok())());
}

}  // namespace test

}  // namespace oneflow
