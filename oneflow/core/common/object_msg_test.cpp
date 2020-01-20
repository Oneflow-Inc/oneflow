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
  foo.__InitRefCount__();
  foo.__IncreaseRefCount__();
  foo.__IncreaseRefCount__();
  ASSERT_EQ(foo.__DecreaseRefCount__(), 1);
  ASSERT_EQ(foo.__DecreaseRefCount__(), 0);
}

TEST(ObjectMsgPtr, new_delete) {
  std::string str;
  {
    class Foo final : public ObjectMsgStruct {
     public:
      void __Delete__() { *str_ = "__Delete__"; }

     private:
      std::string* str_;
    };
    auto foo = ObjectMsgPtr<Foo>::New();
    foo->str_ = &str;
  }
  ASSERT_TRUE(str == "__Delete__");
}

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgFoo)
 public:
  void __Delete__();

  OBJECT_MSG_DEFINE_FIELD(int8_t, x);
  OBJECT_MSG_DEFINE_FIELD(int32_t, foo);
  OBJECT_MSG_DEFINE_FIELD(int16_t, bar);
  OBJECT_MSG_DEFINE_FIELD(int64_t, foobar);
  OBJECT_MSG_DEFINE_RAW_PTR_FIELD(std::string*, is_deleted);
END_OBJECT_MSG(ObjectMsgFoo)
// clang-format on

void OBJECT_MSG_TYPE(ObjectMsgFoo)::__Delete__() {
  if (mutable_is_deleted()) { *mutable_is_deleted() = "deleted"; }
}

TEST(OBJECT_MSG, naive) {
  auto foo = OBJECT_MSG_PTR(ObjectMsgFoo)::New();
  foo->set_bar(9527);
  ASSERT_TRUE(foo->bar() == 9527);
}

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgBar)
  OBJECT_MSG_DEFINE_FIELD(ObjectMsgFoo, bar);
END_OBJECT_MSG(ObjectMsgBar)
// clang-format on

/*
TEST(OBJECT_MSG, nested_objects) {
  //  auto bar = OBJECT_MSG_PTR(ObjectMsgBar)::New();
  //  bar->mutable_bar()->set_bar(9527);
  //  ASSERT_TRUE(bar->bar()->bar() == 9527);
}

TEST(OBJECT_MSG, objects_gc) {
  //  std::string str;
  //  {
  //    auto bar = OBJECT_MSG_PTR(ObjectMsgBar)::New();
  //    bar->mutable_bar()->set_is_deleted(&str);
  //  }
  //  ASSERT_TRUE(str == "deleted");
}
*/
}  // namespace test

}  // namespace oneflow
