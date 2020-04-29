// include sstream first to avoid some compiling error
// caused by the following trick
// reference: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65899
#include <sstream>
#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace test {

TEST(ObjectMsgStruct, ref_cnt) {
  class Foo final : public ObjectMsgBase {
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

}  // namespace test

}  // namespace oneflow
