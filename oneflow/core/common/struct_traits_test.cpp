#include "oneflow/core/common/struct_traits.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

struct OneflowTestNamespaceFoo {
  OneflowTestNamespaceFoo() : x(0), bar(0), const_bar(0) {}

  int x;
  int bar;
  const int const_bar;
};

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

struct X {
  int a;
  int b;
};

struct Y {
  int c;
  X d;
};

TEST(StructField, compose) {
  using BFieldInY = typename ComposeStructField<STRUCT_FIELD(Y, d), STRUCT_FIELD(X, b)>::type;
  Y y;
  int* field_b = &y.d.b;
  ASSERT_EQ(BFieldInY::FieldPtr4StructPtr(&y), field_b);
  ASSERT_EQ(BFieldInY::StructPtr4FieldPtr(field_b), &y);
}

}  // namespace

}  // namespace test

}  // namespace oneflow
