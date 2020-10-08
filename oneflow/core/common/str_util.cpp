/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <glog/logging.h>
#include "oneflow/core/common/str_util.h"

namespace oneflow {

const char* StrToToken(const char* text, const std::string& delims, std::string* token) {
  token->clear();
  while (*text != '\0' && delims.find(*text) != std::string::npos) { text++; }
  while (*text != '\0' && delims.find(*text) == std::string::npos) { token->push_back(*text++); }
  return text;
}

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

std::string Basename(const std::string& path) {
  size_t found = path.rfind('/');
  if (found == std::string::npos) { return path; }
  return path.substr(found + 1);
}

std::string CleanPath(const std::string& unclean_path) {
  std::string path = unclean_path;
  const char* src = path.c_str();
  std::string::iterator dst = path.begin();

  // Check for absolute path and determine initial backtrack limit.
  const bool is_absolute_path = *src == '/';
  if (is_absolute_path) {
    *dst++ = *src++;
    while (*src == '/') ++src;
  }
  std::string::const_iterator backtrack_limit = dst;

  // Process all parts
  while (*src) {
    bool parsed = false;

    if (src[0] == '.') {
      //  1dot ".<whateverisnext>", check for END or SEP.
      if (src[1] == '/' || !src[1]) {
        if (*++src) { ++src; }
        parsed = true;
      } else if (src[1] == '.' && (src[2] == '/' || !src[2])) {
        // 2dot END or SEP (".." | "../<whateverisnext>").
        src += 2;
        if (dst != backtrack_limit) {
          // We can backtrack the previous part
          for (--dst; dst != backtrack_limit && dst[-1] != '/'; --dst) {
            // Empty.
          }
        } else if (!is_absolute_path) {
          // Failed to backtrack and we can't skip it either. Rewind and copy.
          src -= 2;
          *dst++ = *src++;
          *dst++ = *src++;
          if (*src) { *dst++ = *src; }
          // We can never backtrack over a copied "../" part so set new limit.
          backtrack_limit = dst;
        }
        if (*src) { ++src; }
        parsed = true;
      }
    }

    // If not parsed, copy entire part until the next SEP or EOS.
    if (!parsed) {
      while (*src && *src != '/') { *dst++ = *src++; }
      if (*src) { *dst++ = *src++; }
    }

    // Skip consecutive SEP occurrences
    while (*src == '/') { ++src; }
  }

  // Calculate and check the length of the cleaned path.
  std::string::difference_type path_length = dst - path.begin();
  if (path_length != 0) {
    // Remove trailing '/' except if it is root path ("/" ==> path_length := 1)
    if (path_length > 1 && path[path_length - 1] == '/') { --path_length; }
    path.resize(path_length);
  } else {
    // The cleaned path is empty; assign "." as per the spec.
    path.assign(1, '.');
  }
  return path;
}

void GetPrefixAndIndex(const std::string& prefix_and_idx, std::string* prefix, int32_t* index) {
  const size_t underline_pos = prefix_and_idx.rfind('_');
  CHECK_NE(underline_pos, std::string::npos);
  CHECK_GT(underline_pos, 0);
  CHECK_LT(underline_pos, prefix_and_idx.size() - 1);
  *prefix = prefix_and_idx.substr(0, underline_pos);
  *index = oneflow_cast<int32_t>(prefix_and_idx.substr(underline_pos + 1));
  CHECK_GE(*index, 0);
}

bool TryGetPrefixAndIndex(const std::string& prefix_and_idx, std::string* prefix, int32_t* index) {
  const size_t underline_pos = prefix_and_idx.rfind('_');
  if (underline_pos == std::string::npos) { return false; }
  if (underline_pos == 0) { return false; }
  if (underline_pos == prefix_and_idx.size() - 1) { return false; }
  *prefix = prefix_and_idx.substr(0, underline_pos);
  std::string index_str = prefix_and_idx.substr(underline_pos + 1);
  if (IsStrInt(index_str) == false) { return false; }
  *index = oneflow_cast<int32_t>(index_str);
  return *index >= 0;
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

std::string GetHashKeyImpl(std::initializer_list<int> integers) {
  std::string result = "";
  for (int integer : integers) { result += std::to_string(integer) + ","; }
  return result;
}

}  // namespace internal

}  // namespace oneflow
