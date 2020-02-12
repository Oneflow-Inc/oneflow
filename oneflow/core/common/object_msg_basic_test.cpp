#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace test {

TEST(ObjectMsgStruct, ref_cnt) {
  class Foo final : public ObjectMsgStruct {
   public:
    Foo() = default;
  };
  Foo foo;
  foo.InitRefCount();
  foo.IncreaseRefCount();
  foo.IncreaseRefCount();
  ASSERT_EQ(foo.DecreaseRefCount(), 1);
  ASSERT_EQ(foo.DecreaseRefCount(), 0);
}

class TestNew final : public ObjectMsgStruct {
 public:
  static const bool __is_object_message_type__ = true;
  BEGIN_DSS(DSS_GET_FIELD_COUNTER(), TestNew, sizeof(ObjectMsgStruct));
  OBJECT_MSG_DEFINE_INIT();
  OBJECT_MSG_DEFINE_DELETE();

  END_DSS(DSS_GET_FIELD_COUNTER(), "object_msg", TestNew);
};

TEST(ObjectMsgPtr, obj_new) { ObjectMsgPtr<TestNew>::New(); }

}  // namespace test

}  // namespace oneflow
