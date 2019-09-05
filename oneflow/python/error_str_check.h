#include "oneflow/core/common/maybe.h"

std::string OFStrCat() { return Error::CheckFailed(); }
template<typename T0>
std::string OFStrCat(T0 t0) {
  return Error::CheckFailed() << t0;
}
template<typename T0, typename T1>
std::string OFStrCat(T0 t0, T1 t1) {
  return Error::CheckFailed() << t0 << t1;
}
template<typename T0, typename T1, typename T2>
std::string OFStrCat(T0 t0, T1 t1, T2 t2) {
  return Error::CheckFailed() << t0 << t1 << t2;
}
template<typename T0, typename T1, typename T2, typename T3>
std::string OFStrCat(T0 t0, T1 t1, T2 t2, T3 t3) {
  return Error::CheckFailed() << t0 << t1 << t2 << t3;
}
template<typename T0, typename T1, typename T2, typename T3, typename T4>
std::string OFStrCat(T0 t0, T1 t1, T2 t2, T3 t3, T4 t4) {
  return Error::CheckFailed() << t0 << t1 << t2 << t3 << t4;
}

#define OF_ERROR_STR_CHECK(expr, ret_val, ...) \
  do {                                         \
    if (!expr) {                               \
      *error_str = OFStrCat(##__VA_ARGS__);    \
      return ret_val;                          \
    }                                          \
  } while (0)

#define OF_ERROR_STR_CHECK_ISNULL(expr, ret_val, ...) \
  OF_ERROR_STR_CHECK(expr == nullptr, ret_val, __VA_ARGS__)

#define OF_ERROR_STR_CHECK_NOTNULL(expr, ret_val, ...) \
  OF_ERROR_STR_CHECK(expr != nullptr, ret_val, __VA_ARGS__)
