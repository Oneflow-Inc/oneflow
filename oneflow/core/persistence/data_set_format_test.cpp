#include "oneflow/core/persistence/data_set_format.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

TEST(data_set_format, flexible_naive) {
#define CHECK_BASIC_DATA_TYPE_SIZE_OF(type, type_case) \
  ASSERT_EQ(sizeof(type), FlexibleSizeOf<type>(0));

  OF_PP_FOR_EACH_TUPLE(CHECK_BASIC_DATA_TYPE_SIZE_OF, ALL_DATA_TYPE_SEQ);
}

}  // namespace oneflow
