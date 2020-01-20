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
  BEGIN_DSS(Foo, 0);
  int x;
  int y;
  int z;

  DSS_DEFINE_FIELD("demo dss", Foo, x);
  DSS_DEFINE_FIELD("demo dss", Foo, y);
  DSS_DEFINE_FIELD("demo dss", Foo, z);

  END_DSS("demo dss", Foo);
};

struct Bar {
  BEGIN_DSS(Foo, 0);

  END_DSS("demo dss", Bar);
};

template<typename T>
struct IsPointer {
  static const bool value = std::is_pointer<T>::value;
};

template<typename T>
struct RemovePointer {
  using type = typename std::remove_pointer<T>::type;
};

template<typename T>
struct IsScalar {
  static const bool value =
      std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_same<T, std::string>::value;
};

struct FooBar {
  DEFINE_GETTER(int, x);
  DEFINE_MUTABLE(int, x);
  DEFINE_SETTER(IsScalar, int, x);

  DEFINE_GETTER(Foo*, foo);
  DEFINE_MUTABLE(Foo*, foo);
  DEFINE_SETTER(IsScalar, Foo*, foo);

  int x_;
  Foo* foo_;
};

template<typename WalkCtxType, typename FieldType>
struct DumpFieldName {
  static void Call(WalkCtxType* ctx, FieldType* field, const char* field_name) {
    ctx->push_back(field_name);
  }
};

TEST(DSS, walk_field) {
  Foo foo;
  std::vector<std::string> field_names;
  foo.__WalkField__<DumpFieldName>(&field_names);
  ASSERT_EQ(field_names.size(), 3);
  ASSERT_TRUE(field_names[0] == "x");
  ASSERT_TRUE(field_names[1] == "y");
  ASSERT_TRUE(field_names[2] == "z");
}

TEST(DSS, reverse_walk_field) {
  Foo foo;
  std::vector<std::string> field_names;
  foo.__ReverseWalkField__<DumpFieldName>(&field_names);
  ASSERT_EQ(field_names.size(), 3);
  ASSERT_TRUE(field_names[0] == "z");
  ASSERT_TRUE(field_names[1] == "y");
  ASSERT_TRUE(field_names[2] == "x");
}

}  // namespace

}  // namespace oneflow
