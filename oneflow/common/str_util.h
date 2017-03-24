#ifndef _COMMON_STR_UTIL_H_
#define _COMMON_STR_UTIL_H_
#include <string>
#include <vector>
namespace oneflow {
namespace strings {
bool StartsWith(const std::string& source, const std::string& search_for);
bool EndsWith(const std::string& source, const std::string& search_for);
bool Contains(const std::string& source, const std::string& search_for);
std::vector<std::string> Split(const std::string& source, 
  const std::string& delim);
std::string Join(const std::vector<std::string>& source,
  const std::string& deilm);
std::string RemoveSuffix(const std::string& source, const std::string& suffix);
std::string RemovePrefix(const std::string& source, const std::string& prefix);
}  // namespace strings
}  // namespace oneflow
#endif  // _COMMON_STR_UTIL_H_
