#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

bool IsIntegralDataType(DataType data_type) {
  switch (data_type) {
#define INTERGRAL_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(INTERGRAL_CASE, INT_DATA_TYPE_SEQ)
    default: return false;
  }
}
bool IsFloatingDataType(DataType data_type) {
  switch (data_type) {
#define FLOATING_CASE(type_cpp, type_proto) \
  case type_proto: return true;
    OF_PP_FOR_EACH_TUPLE(FLOATING_CASE, FLOATING_DATA_TYPE_SEQ)
    default: return false;
  }
}

size_t GetSizeOfDataType(DataType data_type) {
  switch (data_type) {
#define MAKE_CASE(type_cpp, type_proto) \
  case type_proto: return sizeof(type_cpp);
    OF_PP_FOR_EACH_TUPLE(MAKE_CASE,
                         ALL_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
    default: UNIMPLEMENTED();
  }
}

namespace {

void CheckDataType() {
  static_assert(sizeof(int8_t) == sizeof(char), "sizeof(int8_t) != sizeof(char)");
  static_assert(sizeof(int16_t) == sizeof(short), "sizeof(int16_t) != sizeof(short)");
  static_assert(sizeof(int32_t) == sizeof(int), "sizeof(int32_t) != sizeof(int)");
  static_assert(sizeof(int64_t) == sizeof(long long), "sizeof(int64_t) != sizeof(long long)");

  CHECK(GetZeroVal<float16>() == float16(0.0));
  CHECK(GetOneVal<float16>() == float16(1.0));
  CHECK(GetMaxVal<float16>() == std::numeric_limits<float16>::max());
  CHECK(GetMinVal<float16>() == std::numeric_limits<float16>::lowest());

#define CHECK_MAX_VAL(T, limit_value) CHECK_EQ(GetMaxVal<T>(), std::numeric_limits<T>::max());
  OF_PP_FOR_EACH_TUPLE(CHECK_MAX_VAL, MAX_VAL_SEQ);
#undef CHECK_MAX_VAL

#define CHECK_MIN_VAL(T, limit_value) CHECK_EQ(GetMinVal<T>(), std::numeric_limits<T>::lowest());
  OF_PP_FOR_EACH_TUPLE(CHECK_MIN_VAL, MIN_VAL_SEQ);
#undef CHECK_MIN_VAL
}

COMMAND(CheckDataType());

}  // namespace

}  // namespace oneflow
