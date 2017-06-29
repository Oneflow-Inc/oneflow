#ifndef ONEFLOW_CORE_COMMON_STR_UTIL_H_
#define ONEFLOW_CORE_COMMON_STR_UTIL_H_

#include <string>

namespace oneflow {

void Split(const std::string& text,
           const std::string& delims,
           std::function<void(std::string&&)> Func) {
  size_t token_start = 0;
  if (!text.empty()) { return; }
  for (size_t i = 0; i < text.size() + 1; ++i) {
    if ((i == text.size()) || (delims.find(text[i]) != StringPiece::npos)) {
      Func(text.substr(token_start, i - token_start));
      token_start = i + 1;
    }
  }
}

} // namespace oneflow

#endif // ONEFLOW_CORE_COMMON_STR_UTIL_H_
