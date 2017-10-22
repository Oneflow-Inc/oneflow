#include "oneflow/core/common/flexible.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {
namespace test {

TEST(flexible, basic_data_type) {
#define CHECK_BASIC_DATA_TYPE_SIZE_OF(type, type_case) \
  ASSERT_EQ(sizeof(type), Flexible<type>::SizeOf(0));
  OF_PP_FOR_EACH_TUPLE(CHECK_BASIC_DATA_TYPE_SIZE_OF, ALL_DATA_TYPE_SEQ);
}

}  // namespace test
}  // namespace oneflow
