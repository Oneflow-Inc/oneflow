#ifndef ONEFLOW_CORE_PLATFORM_STRINGPRINTF_H_
#define ONEFLOW_CORE_PLATFORM_STRINGPRINTF_H_

#define OF_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))

#include "oneflow/core/framework/framework.h"
#include <stdarg.h>

namespace oneflow {

namespace strings {

extern std::string Printf(const char* format, ...) OF_PRINTF_ATTRIBUTE(1, 2);

extern void Appendf(std::string* dst, const char* format, ...) OF_PRINTF_ATTRIBUTE(2, 3);

extern void Appendv(std::string* dst, const char* format, va_list ap);

}  // namespace strings
}  // namespace oneflow

#endif