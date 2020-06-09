#include "oneflow/customized/utils/stringprintf.h"
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>

namespace oneflow {
namespace strings {

void Appendv(std::string* dst, const char* format, va_list ap) {
  static const int kSpaceLength = 1024;
  char space[kSpaceLength];

  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(space, kSpaceLength, format, backup_ap);
  va_end(backup_ap);

  if (result < kSpaceLength) {
    if (result >= 0) {
      dst->append(space, result);
      return;
    }
#ifdef _MSC_VER
    va_copy(backup_ap, ap);
    result = vsnprintf(nullptr, 0, format, backup_ap);
    va_end(backup_ap);
#endif

    if (result < 0) { return; }
  }

  int length = result + 1;
  char* buf = new char[length];
  va_copy(backup_ap, ap);
  result = vsnprintf(buf, length, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < length) { dst->append(buf, result); }
  delete[] buf;
}

std::string Printf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  Appendv(&result, format, ap);
  va_end(ap);
  return result;
}

void Appendf(std::string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  Appendv(dst, format, ap);
  va_end(ap);
}

}  // namespace strings
}  // namespace oneflow
