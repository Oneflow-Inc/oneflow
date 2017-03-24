#include "common/str_util.h"
#include <regex>
#include <sstream>
#include <ostream>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_set>
#include <glog/logging.h>

namespace caffe {
namespace strings {

bool StartsWith(const std::string& source, const std::string& search_for) {
  if (search_for.size() > source.size()) return false;
  std::string source_str = source.substr(0, search_for.size());
  return source_str == search_for;
}

bool EndsWith(const std::string& source, const std::string& search_for) {
  if (search_for.size() > source.size()) return false;
  std::string source_for
    = source.substr(source.size() - search_for.size(), search_for.size());
  return source_for == search_for;
}

bool Contains(const std::string& source, const std::string& search_for) {
  if (search_for.size() > source.size()) return false;
  return (source.find(search_for) != std::string::npos);
}

std::vector<std::string> Split(const std::string& source,
  const std::string& delim) {
  std::regex re(delim);
  std::sregex_token_iterator begin{ source.begin(), source.end(), re, -1 };
  std::sregex_token_iterator end;
  return {begin, end};
}

std::string Join(const std::vector<std::string>& source,
  const std::string& delim) {
  std::stringstream ss;
  std::copy(source.begin(), source.end(),
    std::ostream_iterator<std::string>(ss, delim.c_str()));
  auto join_str = ss.str();
  return join_str.substr(0, join_str.length() - delim.length());
}

std::string RemoveSuffix(const std::string& source, const std::string& suffix) {
  auto pos = source.rfind(suffix);
  CHECK(pos != std::string::npos);
  return source.substr(0, pos);
}

std::string RemovePrefix(const std::string& source, const std::string& prefix) {
  CHECK(StartsWith(source, prefix));
  return source.substr(prefix.length());
}

bool has_diff_correspondence(const std::vector<std::string>& blob_names,
  const std::vector<std::string>& diff_names) {
  std::unordered_set<std::string> diff_set;
  for (auto& diff_name : diff_names) {
    diff_set.insert(diff_name);
  }
  // Return true once one of the blob has its diff correspondence
  for (auto& blob_name : blob_names) {
    auto diff_name = strings::Join({ blob_name, "diff" }, "_");
    if (diff_set.count(diff_name)) {
      return true;
    }
  }
  return false;
}

}  // namespace strings
}  // namespace caffe