/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/lib/strings/numbers.h"

#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

template <typename T>
T locale_independent_strtonum(const char* str, const char** endptr) {
  static const std::unordered_map<string, T> special_nums = {
      {"inf", std::numeric_limits<T>::infinity()},
      {"+inf", std::numeric_limits<T>::infinity()},
      {"-inf", -std::numeric_limits<T>::infinity()},
      {"infinity", std::numeric_limits<T>::infinity()},
      {"+infinity", std::numeric_limits<T>::infinity()},
      {"-infinity", -std::numeric_limits<T>::infinity()},
      {"nan", std::numeric_limits<T>::quiet_NaN()},
      {"+nan", std::numeric_limits<T>::quiet_NaN()},
      {"-nan", -std::numeric_limits<T>::quiet_NaN()},
  };
  std::stringstream s(str);

  // Check if str is one of the special numbers.
  string special_num_str;
  s >> special_num_str;

  for (int i = 0; i < special_num_str.length(); ++i) {
    special_num_str[i] =
        std::tolower(special_num_str[i], std::locale::classic());
  }

  auto entry = special_nums.find(special_num_str);
  if (entry != special_nums.end()) {
    *endptr = str + (s.eof() ? static_cast<std::iostream::pos_type>(strlen(str))
                             : s.tellg());
    return entry->second;
  } else {
    // Perhaps it's a hex number
    if (special_num_str.compare(0, 2, "0x") == 0 ||
        special_num_str.compare(0, 3, "-0x") == 0) {
      return strtol(str, const_cast<char**>(endptr), 16);
    }
  }
  // Reset the stream
  s.str(str);
  s.clear();
  // Use the "C" locale
  s.imbue(std::locale::classic());

  T result;
  s >> result;

  // Set to result to what strto{f,d} functions would have returned. If the
  // number was outside the range, the stringstream sets the fail flag, but
  // returns the +/-max() value, whereas strto{f,d} functions return +/-INF.
  bool real_fail = false;
  if (s.fail()) {
    real_fail = true;
    if (result == std::numeric_limits<T>::max()) {
      result = std::numeric_limits<T>::infinity();
      real_fail = false;
    } else if (result == -std::numeric_limits<T>::max()) {
      result = -std::numeric_limits<T>::infinity();
      real_fail = false;
    }
  }

  if (endptr) {
    *endptr =
        str +
        (real_fail
             ? static_cast<std::iostream::pos_type>(0)
             : (s.eof() ? static_cast<std::iostream::pos_type>(strlen(str))
                        : s.tellg()));
  }
  return result;
}

}  // namespace

namespace strings {

char* FastInt32ToBufferLeft(int32 i, char* buffer) {
  uint32 u = i;
  if (i < 0) {
    *buffer++ = '-';
    // We need to do the negation in modular (i.e., "unsigned")
    // arithmetic; MSVC++ apparently warns for plain "-u", so
    // we write the equivalent expression "0 - u" instead.
    u = 0 - u;
  }
  return FastUInt32ToBufferLeft(u, buffer);
}

char* FastUInt32ToBufferLeft(uint32 i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = 0;
  std::reverse(start, buffer);
  return buffer;
}

char* FastInt64ToBufferLeft(int64 i, char* buffer) {
  uint64 u = i;
  if (i < 0) {
    *buffer++ = '-';
    u = 0 - u;
  }
  return FastUInt64ToBufferLeft(u, buffer);
}

char* FastUInt64ToBufferLeft(uint64 i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = 0;
  std::reverse(start, buffer);
  return buffer;
}

static const double kDoublePrecisionCheckMax = DBL_MAX / 1.000000000000001;

char* DoubleToBuffer(double value, char* buffer) {
  // DBL_DIG is 15 for IEEE-754 doubles, which are used on almost all
  // platforms these days.  Just in case some system exists where DBL_DIG
  // is significantly larger -- and risks overflowing our buffer -- we have
  // this assert.
  static_assert(DBL_DIG < 20, "DBL_DIG is too big");

  bool full_precision_needed = true;
  if (std::abs(value) <= kDoublePrecisionCheckMax) {
    int snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG, value);

    // The snprintf should never overflow because the buffer is significantly
    // larger than the precision we asked for.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

    full_precision_needed =
        locale_independent_strtonum<double>(buffer, NULL) != value;
  }

  if (full_precision_needed) {
    int snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG + 2, value);

    // Should never overflow; see above.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
  }
  return buffer;
}

namespace {
char SafeFirstChar(StringPiece str) {
  if (str.empty()) return '\0';
  return str[0];
}
void SkipSpaces(StringPiece* str) {
  while (isspace(SafeFirstChar(*str))) str->remove_prefix(1);
}
}  // namespace

bool safe_strto64(StringPiece str, int64* value) {
  SkipSpaces(&str);

  int64 vlimit = kint64max;
  int sign = 1;
  if (str.Consume("-")) {
    sign = -1;
    // Different limit for positive and negative integers.
    vlimit = kint64min;
  }

  if (!isdigit(SafeFirstChar(str))) return false;

  int64 result = 0;
  if (sign == 1) {
    do {
      int digit = SafeFirstChar(str) - '0';
      if ((vlimit - digit) / 10 < result) {
        return false;
      }
      result = result * 10 + digit;
      str.remove_prefix(1);
    } while (isdigit(SafeFirstChar(str)));
  } else {
    do {
      int digit = SafeFirstChar(str) - '0';
      if ((vlimit + digit) / 10 > result) {
        return false;
      }
      result = result * 10 - digit;
      str.remove_prefix(1);
    } while (isdigit(SafeFirstChar(str)));
  }

  SkipSpaces(&str);
  if (!str.empty()) return false;

  *value = result;
  return true;
}

bool safe_strtou64(StringPiece str, uint64* value) {
  SkipSpaces(&str);
  if (!isdigit(SafeFirstChar(str))) return false;

  int64 result = 0;
  do {
    int digit = SafeFirstChar(str) - '0';
    if ((kuint64max - digit) / 10 < result) {
      return false;
    }
    result = result * 10 + digit;
    str.remove_prefix(1);
  } while (isdigit(SafeFirstChar(str)));

  SkipSpaces(&str);
  if (!str.empty()) return false;

  *value = result;
  return true;
}

bool safe_strto32(StringPiece str, int32* value) {
  SkipSpaces(&str);

  int64 vmax = kint32max;
  int sign = 1;
  if (str.Consume("-")) {
    sign = -1;
    // Different max for positive and negative integers.
    ++vmax;
  }

  if (!isdigit(SafeFirstChar(str))) return false;

  int64 result = 0;
  do {
    result = result * 10 + SafeFirstChar(str) - '0';
    if (result > vmax) {
      return false;
    }
    str.remove_prefix(1);
  } while (isdigit(SafeFirstChar(str)));

  SkipSpaces(&str);

  if (!str.empty()) return false;

  *value = result * sign;
  return true;
}

bool safe_strtou32(StringPiece str, uint32* value) {
  SkipSpaces(&str);
  if (!isdigit(SafeFirstChar(str))) return false;

  int64 result = 0;
  do {
    result = result * 10 + SafeFirstChar(str) - '0';
    if (result > kuint32max) {
      return false;
    }
    str.remove_prefix(1);
  } while (isdigit(SafeFirstChar(str)));

  SkipSpaces(&str);
  if (!str.empty()) return false;

  *value = result;
  return true;
}

bool safe_strtof(const char* str, float* value) {
  const char* endptr;
  *value = locale_independent_strtonum<float>(str, &endptr);
  while (isspace(*endptr)) ++endptr;
  // Ignore range errors from strtod/strtof.
  // The values it returns on underflow and
  // overflow are the right fallback in a
  // robust setting.
  return *str != '\0' && *endptr == '\0';
}

bool safe_strtod(const char* str, double* value) {
  const char* endptr;
  *value = locale_independent_strtonum<double>(str, &endptr);
  while (isspace(*endptr)) ++endptr;
  // Ignore range errors from strtod/strtof.
  // The values it returns on underflow and
  // overflow are the right fallback in a
  // robust setting.
  return *str != '\0' && *endptr == '\0';
}

char* FloatToBuffer(float value, char* buffer) {
  // FLT_DIG is 6 for IEEE-754 floats, which are used on almost all
  // platforms these days.  Just in case some system exists where FLT_DIG
  // is significantly larger -- and risks overflowing our buffer -- we have
  // this assert.
  static_assert(FLT_DIG < 10, "FLT_DIG is too big");

  int snprintf_result =
      snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG, value);

  // The snprintf should never overflow because the buffer is significantly
  // larger than the precision we asked for.
  DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

  float parsed_value;
  if (!safe_strtof(buffer, &parsed_value) || parsed_value != value) {
    snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG + 2, value);

    // Should never overflow; see above.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
  }
  return buffer;
}

string FpToString(Fprint fp) {
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<uint64>(fp));
  return string(buf);
}

bool StringToFp(const string& s, Fprint* fp) {
  char junk;
  uint64 result;
  if (sscanf(s.c_str(), "%llx%c", &result, &junk) == 1) {
    *fp = result;
    return true;
  } else {
    return false;
  }
}

StringPiece Uint64ToHexString(uint64 v, char* buf) {
  static const char* hexdigits = "0123456789abcdef";
  const int num_byte = 16;
  buf[num_byte] = '\0';
  for (int i = num_byte - 1; i >= 0; i--) {
    buf[i] = hexdigits[v & 0xf];
    v >>= 4;
  }
  return StringPiece(buf, num_byte);
}

bool HexStringToUint64(const StringPiece& s, uint64* result) {
  uint64 v = 0;
  if (s.empty()) {
    return false;
  }
  for (size_t i = 0; i < s.size(); i++) {
    char c = s[i];
    if (c >= '0' && c <= '9') {
      v = (v << 4) + (c - '0');
    } else if (c >= 'a' && c <= 'f') {
      v = (v << 4) + 10 + (c - 'a');
    } else if (c >= 'A' && c <= 'F') {
      v = (v << 4) + 10 + (c - 'A');
    } else {
      return false;
    }
  }
  *result = v;
  return true;
}

string HumanReadableNumBytes(int64 num_bytes) {
  if (num_bytes == kint64min) {
    // Special case for number with not representable negation.
    return "-8E";
  }

  const char* neg_str = (num_bytes < 0) ? "-" : "";
  if (num_bytes < 0) {
    num_bytes = -num_bytes;
  }

  // Special case for bytes.
  if (num_bytes < 1024) {
    // No fractions for bytes.
    char buf[8];  // Longest possible string is '-XXXXB'
    snprintf(buf, sizeof(buf), "%s%lldB", neg_str,
             static_cast<int64>(num_bytes));
    return string(buf);
  }

  static const char units[] = "KMGTPE";  // int64 only goes up to E.
  const char* unit = units;
  while (num_bytes >= static_cast<int64>(1024) * 1024) {
    num_bytes /= 1024;
    ++unit;
    CHECK(unit < units + TF_ARRAYSIZE(units));
  }

  // We use SI prefixes.
  char buf[16];
  snprintf(buf, sizeof(buf), ((*unit == 'K') ? "%s%.1f%ciB" : "%s%.2f%ciB"),
           neg_str, num_bytes / 1024.0, *unit);
  return string(buf);
}

}  // namespace strings
}  // namespace tensorflow
