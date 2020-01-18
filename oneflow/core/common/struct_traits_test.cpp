#include "oneflow/core/common/struct_traits.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

struct OneflowTestNamespaceFoo {
  OneflowTestNamespaceFoo() : x(0), bar(0), const_bar(0) {}

  int x;
  int bar;
  const int const_bar;
};

DEFINE_STRUCT_FIELD(OneflowTestNamespaceFoo, bar);
DEFINE_STRUCT_FIELD(OneflowTestNamespaceFoo, const_bar);

TEST(StructField, mutable_struct_mutable_field) {
  OneflowTestNamespaceFoo foo;
  auto* bar = &foo.bar;
  auto* struct_ptr = STRUCT_FIELD(OneflowTestNamespaceFoo, bar)::StructPtr4FieldPtr(bar);
  auto* field_ptr = STRUCT_FIELD(OneflowTestNamespaceFoo, bar)::FieldPtr4StructPtr(&foo);
  ASSERT_EQ(struct_ptr, &foo);
  ASSERT_EQ(field_ptr, bar);
}

TEST(StructField, mutable_struct_const_field) {
  OneflowTestNamespaceFoo foo;
  auto* bar = &foo.const_bar;
  auto* struct_ptr = STRUCT_FIELD(OneflowTestNamespaceFoo, const_bar)::StructPtr4FieldPtr(bar);
  auto* field_ptr = STRUCT_FIELD(OneflowTestNamespaceFoo, const_bar)::FieldPtr4StructPtr(&foo);
  ASSERT_EQ(struct_ptr, &foo);
  ASSERT_EQ(field_ptr, bar);
}

TEST(StructField, const_struct_mutable_field) {
  const OneflowTestNamespaceFoo foo;
  auto* bar = &foo.bar;
  auto* struct_ptr = STRUCT_FIELD(OneflowTestNamespaceFoo, bar)::StructPtr4FieldPtr(bar);
  auto* field_ptr = STRUCT_FIELD(OneflowTestNamespaceFoo, bar)::FieldPtr4StructPtr(&foo);
  ASSERT_EQ(struct_ptr, &foo);
  ASSERT_EQ(field_ptr, bar);
}

TEST(StructField, const_struct_const_field) {
  const OneflowTestNamespaceFoo foo;
  auto* bar = &foo.const_bar;
  auto* struct_ptr = STRUCT_FIELD(OneflowTestNamespaceFoo, const_bar)::StructPtr4FieldPtr(bar);
  auto* field_ptr = STRUCT_FIELD(OneflowTestNamespaceFoo, const_bar)::FieldPtr4StructPtr(&foo);
  ASSERT_EQ(struct_ptr, &foo);
  ASSERT_EQ(field_ptr, bar);
}

namespace {

struct Foo {
  DSS_DECLARE_CODE_LINE_FIELD_SIZE_AND_OFFSET();
  DSS_CHECK_CODE_LINE_FIELD_SIZE_AND_OFFSET("demo dss", sizeof(((Foo*)nullptr)->x),
                                            &((Foo*)nullptr)->x);
  DSS_CHECK_CODE_LINE_FIELD_SIZE_AND_OFFSET("demo dss", sizeof(((Foo*)nullptr)->y),
                                            &((Foo*)nullptr)->y);
  DSS_CHECK_CODE_LINE_FIELD_SIZE_AND_OFFSET("demo dss", sizeof(((Foo*)nullptr)->z),
                                            &((Foo*)nullptr)->z);

  int x;
  int y;
  int z;

  DSS_STATIC_ASSERT_STRUCT_SIZE("demo dss", Foo);
};

}  // namespace

}  // namespace oneflow
