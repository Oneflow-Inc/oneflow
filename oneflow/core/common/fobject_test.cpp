#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/fobject.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace test {

TEST(FObjectStruct, ref_cnt) {
  class Foo final : public FObjectStruct {
   public:
    Foo() = default;
  };
  Foo foo;
  foo.__InitRefCount__();
  foo.__IncreaseRefCount__();
  foo.__IncreaseRefCount__();
  ASSERT_EQ(foo.__DecreaseRefCount__(), 1);
  ASSERT_EQ(foo.__DecreaseRefCount__(), 0);
}

TEST(FObject, new_delete) {
  std::string str;
  {
    class Foo final : public FObjectStruct {
     public:
      void __Init__(std::string* str) { str_ = str; }
      void __Delete__() { *str_ = "__Delete__"; }

     private:
      std::string* str_;
    };
    auto foo = FObject<Foo>::New(&str);
  }
  ASSERT_TRUE(str == "__Delete__");
}

// clang-format off
BEGIN_FOBJECT(FObjectFoo)
 public:
  void __Init__(int32_t bar);

  FOBJECT_DEFINE_FIELD(int8_t, x);
  FOBJECT_DEFINE_FIELD(int32_t, foo);
  FOBJECT_DEFINE_FIELD(int16_t, bar);
  FOBJECT_DEFINE_FIELD(int64_t, foobar);
END_FOBJECT(FObjectFoo)
// clang-format on

void FOBJECT_METHOD(FObjectFoo, __Init__)(int32_t bar) { set_bar(bar); }

TEST(FOBJECT, naive) {
  auto foo = FOBJECT(FObjectFoo)::New(9527);
  ASSERT_TRUE(foo->bar() == 9527);
  //  auto bar = FOBJECT(FObjectBar)::New();
}

// clang-format off
BEGIN_FOBJECT(FObjectBar)
  FOBJECT_DEFINE_FIELD(FObjectFoo, bar);

 public:
  void __Init__();
END_FOBJECT(FObjectBar)
// clang-format on

void FOBJECT_METHOD(FObjectBar, __Init__)() { set_bar(FOBJECT(FObjectFoo)::New(9527)); }

TEST(FOBJECT, nested_objects) {
  auto bar = FOBJECT(FObjectBar)::New();
  ASSERT_TRUE(bar->bar()->bar() == 9527);
}

}  // namespace test

}  // namespace oneflow
