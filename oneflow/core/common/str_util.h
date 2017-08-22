#ifndef ONEFLOW_CORE_COMMON_STR_UTIL_H_
#define ONEFLOW_CORE_COMMON_STR_UTIL_H_

#include <functional>
#include <string>
#include "oneflow/core/common/util.h"

namespace oneflow {

inline void StringReplace(std::string* str, char old_ch, char new_ch) {
  for (size_t i = 0; i < str->size(); ++i) {
    if (str->at(i) == old_ch) { str->at(i) = new_ch; }
  }
}

const char* StrToToken(const char* text, const std::string& delims,
                       std::string* token);

void Split(const std::string& text, const std::string& delims,
           std::function<void(std::string&&)> Func);

template<typename T>
void SplitAndParseAs(const std::string& text, const std::string& delims,
                     std::function<void(T&&)> Func) {
  Split(text, delims, [&Func](std::string&& s) { Func(oneflow_cast<T>(s)); });
}

// Return true if path is absolute.
inline bool IsAbsolutePath(const std::string& path) {
  return !path.empty() && path[0] == '/';
}

namespace internal {

std::string JoinPathImpl(std::initializer_list<std::string> paths);

}  // namespace internal

// Join multiple paths together, without introducing unnecessary path
// separators.
// For example:
//
//  Arguments                  | JoinPath
//  ---------------------------+----------
//  '/foo', 'bar'              | /foo/bar
//  '/foo/', 'bar'             | /foo/bar
//  '/foo', '/bar'             | /foo/bar
//
// Usage:
// string path = JoinPath("/mydir", filename);
// string path = JoinPath(FLAGS_test_srcdir, filename);
// string path = JoinPath("/full", "path", "to", "filename);
template<typename... T>
std::string JoinPath(const T&... args) {
  return internal::JoinPathImpl({args...});
}

// Returns the part of the path before the final "/".  If there is a single
// leading "/" in the path, the result will be the leading "/".  If there is
// no "/" in the path, the result is the empty prefix of the input.
std::string Dirname(const std::string& path);

// Returns the part of the path after the final "/".  If there is no
// "/" in the path, the result is the same as the input.
std::string Basename(const std::string& path);

// Collapse duplicate "/"s, resolve ".." and "." path elements, remove
// trailing "/".
//
// NOTE: This respects relative vs. absolute paths, but does not
// invoke any system calls (getcwd(2)) in order to resolve relative
// paths with respect to the actual working directory.  That is, this is purely
// string manipulation, completely independent of process state.
std::string CleanPath(const std::string& path);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_STR_UTIL_H_
