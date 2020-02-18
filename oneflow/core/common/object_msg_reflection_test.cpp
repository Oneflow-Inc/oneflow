#include "oneflow/core/common/dss.h"
#include "oneflow/core/common/object_msg_reflection.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

#define DSS_DEFINE_TEST_UNION_FIELD(field_counter)                   \
  DSS_DEFINE_FIELD(field_counter, "demo dss", union_field);          \
  DSS_DEFINE_UNION_FIELD_VISITOR(field_counter, union_case,          \
                                 OF_PP_MAKE_TUPLE_SEQ(int32_t, x, 1) \
                                     OF_PP_MAKE_TUPLE_SEQ(int64_t, y, 2));

struct TestDssUnion {
  BEGIN_DSS(DSS_GET_FIELD_COUNTER(), TestDssUnion, 0);

 public:
  struct {
    int32_t union_case;
    union {
      int32_t x;
      int64_t y;
    };
  } union_field;

  DSS_DEFINE_TEST_UNION_FIELD(DSS_GET_FIELD_COUNTER());
  END_DSS(DSS_GET_FIELD_COUNTER(), "demo dss", TestDssUnion);
};

TEST(ObjectMsgReflection, ReflectObjectMsgFields) {
  ObjectMsgFieldList obj_msg_field_list;
  ObjectMsgReflection<TestDssUnion>().ReflectObjectMsgFields(&obj_msg_field_list);
  ASSERT_EQ(obj_msg_field_list.object_msg_field().size(), 1);
  ASSERT_TRUE(obj_msg_field_list.object_msg_field(0).has_union_field_list());
  const auto& union_field_list = obj_msg_field_list.object_msg_field(0).union_field_list();
  ASSERT_EQ(union_field_list.union_name(), "union_field");
  ASSERT_EQ(union_field_list.union_field_name(0), "x");
  ASSERT_EQ(union_field_list.union_field_name(1), "y");
}
}

}  // namespace test

}  // namespace oneflow
