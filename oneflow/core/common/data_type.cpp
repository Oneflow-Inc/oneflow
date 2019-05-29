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

#if defined(WITH_CUDA)

#define CHECK_DEVICE_FP16(get_val)                              \
  do {                                                          \
    float16 host_fp16 = get_val<float16>();                     \
    half device_fp16 = get_val<half>();                         \
    CHECK_EQ(*(uint16_t*)&host_fp16, *(uint16_t*)&device_fp16); \
  } while (0)

  CHECK_DEVICE_FP16(GetZeroVal);
  CHECK_DEVICE_FP16(GetOneVal);
  CHECK_DEVICE_FP16(GetMaxVal);
  CHECK_DEVICE_FP16(GetMinVal);
#undef CHECK_DEVICE_FP16

#endif

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
