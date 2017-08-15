#include "oneflow/core/common/str_util.h"

namespace oneflow {

void Split(const std::string& text, const std::string& delims,
           std::function<void(std::string&&)> Func) {
  size_t token_start = 0;
  if (text.empty()) { return; }
  for (size_t i = 0; i < text.size() + 1; ++i) {
    if ((i == text.size()) || (delims.find(text[i]) != std::string::npos)) {
      Func(text.substr(token_start, i - token_start));
      token_start = i + 1;
    }
  }
}

std::string Dirname(const std::string& path) {
  size_t found = path.rfind('/');
  if (found == std::string::npos) { return ""; }
  if (found == 0) { return "/"; }
  return path.substr(0, found);
}

namespace internal {

std::string JoinPathImpl(std::initializer_list<std::string> paths) {
  std::string result;
  for (std::string path : paths) {
    if (path.empty()) continue;
    if (result.empty()) {
      result = path;
      continue;
    }
    if (result[result.size() - 1] == '/') {
      if (IsAbsolutePath(path)) {
        result.append(path.substr(1));
      } else {
        result.append(path);
      }
    } else {
      if (IsAbsolutePath(path)) {
        result.append(path);
      } else {
        result += ("/" + path);
      }
    }
  }
  return result;
}

}  // namespace internal

}  // namespace oneflow
