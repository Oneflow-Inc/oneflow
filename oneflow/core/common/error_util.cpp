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
#include <sstream>
#include "oneflow/core/common/error_util.h"

namespace oneflow {

namespace {

std::string StripSpace(std::string str) {
  if (str.size() == 0) { return ""; }
  size_t pos = str.find_first_not_of(" ");
  if (pos != std::string::npos) { str.erase(0, pos); }
  pos = str.find_last_not_of(" ") + 1;
  if (pos != std::string::npos) { str.erase(pos); }
  return str;
}

bool IsLetterNumberOrUnderline(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
}

std::string StripBrackets(std::string str) {
  if (str.size() == 0) { return ""; }
  if (str.at(0) != '(') { return str; }
  // "()" come from OF_PP_STRINGIZE((__VA_ARGS__)), so size() >= 2
  str = str.substr(1, str.size() - 2);
  return str;
}

Maybe<std::string> ShortenErrorMsg(std::string str) {
  // 150 characters is the threshold
  const int num_displayed_char = 150;
  if (str.size() == 0) { return str; }
  // strip space when JUST(  xx  );
  str = StripSpace(str);
  if (str.size() < num_displayed_char) { return str; }

  // Find first index where the number of characters from the start to the index is less than 50,
  // last index is the same
  int first_index = -1;
  int last_index = -1;
  int pre_index = 0;
  CHECK_OR_RETURN(str.size() >= 1);
  for (int i = 1; i < str.size(); i++) {
    if (IsLetterNumberOrUnderline(str.at(i)) && !IsLetterNumberOrUnderline(str.at(i - 1))) {
      if (first_index == -1 && i >= num_displayed_char / 3) { first_index = pre_index; }
      if (last_index == -1 && str.size() - i <= num_displayed_char / 3) { last_index = i; }
      pre_index = i;
    }
  }
  // A string of more than 150 characters
  if (first_index == -1 && last_index == -1) { return str; }
  CHECK_OR_RETURN(first_index <= str.size());
  CHECK_OR_RETURN(last_index <= str.size());
  std::stringstream ss;
  // The number of characters before the first word exceeds 50
  if (first_index == -1) {
    ss << " ... " << str.substr(last_index);
  }
  // The number of characters after the last word exceeds 50
  else if (last_index == -1) {
    ss << str.substr(0, first_index) << " ... ";
  } else {
    ss << str.substr(0, first_index) << " ... " << str.substr(last_index);
  }
  return ss.str();
}

std::string FormatFile(const std::string& file) {
  std::stringstream ss;
  ss << "\n  File \"" << file << "\", ";
  return ss.str();
}

std::string FormatLine(const int64_t& line) {
  std::stringstream ss;
  ss << "line " << line << ",";
  return ss.str();
}

std::string FormatFunction(const std::string& function) {
  std::stringstream ss;
  ss << " in " << function;
  return ss.str();
}

Maybe<std::string> FormatErrorMsg(std::string error_msg, bool is_last_stack_frame) {
  error_msg = StripBrackets(error_msg);
  if (!is_last_stack_frame) { error_msg = *JUST(ShortenErrorMsg(error_msg)); }
  // error_msg of last stack frame come from "<<"
  if (is_last_stack_frame) { error_msg = StripSpace(error_msg); }
  std::stringstream ss;
  ss << "\n    " << error_msg;
  return ss.str();
}

std::string FormatErrorSummaryAndMsg(const std::shared_ptr<cfg::ErrorProto>& error) {
  std::stringstream ss;
  if (error->has_error_summary()) { ss << error->error_summary(); }
  if (error->has_msg()) { ss << (ss.str().size() != 0 ? ", " + error->msg() : error->msg()); }
  return ss.str();
}

}  // namespace

Maybe<std::string> FormatErrorStr(const std::shared_ptr<cfg::ErrorProto>& error) {
  std::stringstream ss;
  for (auto stack_frame = error->mutable_stack_frame()->rbegin();
       stack_frame < error->mutable_stack_frame()->rend(); stack_frame++) {
    ss << FormatFile(*stack_frame->mutable_file()) << FormatLine(*stack_frame->mutable_line())
       << FormatFunction(*stack_frame->mutable_function())
       << *JUST(FormatErrorMsg(*stack_frame->mutable_error_msg(),
                               stack_frame == error->mutable_stack_frame()->rend() - 1));
  }
  ss << "\n" << FormatErrorSummaryAndMsg(error);
  return ss.str();
}

}  // namespace oneflow
