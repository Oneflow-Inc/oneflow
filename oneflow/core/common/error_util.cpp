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

std::string* MutErrorStr() {
  thread_local static std::string error_str = "";
  return &error_str;
}

const std::string& GetErrorStr() { return *MutErrorStr(); }

namespace {

void StripSpace(std::string& str) {
  str.erase(0, str.find_first_not_of(" "));
  str.erase(str.find_last_not_of(" ") + 1);
}

bool IsLetterNumberOrUnderline(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
}

void EraseErrorMsg(std::string& str) {
  if (str.size() == 0) { return; }
  StripSpace(str);
  // strip bracket
  if (str.at(0) != '(') { return; }
  str = str.substr(1, str.size() - 2);
}

void ShortenErrorMsg(std::string& str) {
  const int num_displayed_char = 150;
  if (str.size() == 0) { return; }
  StripSpace(str);
  if (str.size() < num_displayed_char) { return; }

  int first_index = -1;
  int last_index = -1;
  int pre_index = 0;
  for (int i = 1; i < str.size(); i++) {
    if (IsLetterNumberOrUnderline(str.at(i)) && !IsLetterNumberOrUnderline(str.at(i - 1))) {
      if (first_index == -1 && i >= num_displayed_char / 3) { first_index = pre_index; }
      if (last_index == -1 && str.size() - i <= num_displayed_char / 3) { last_index = i; }
      pre_index = i;
    }
  }
  std::stringstream ss;
  ss << str.substr(0, first_index) << " ... " << str.substr(last_index);
  str = ss.str();
}

std::string FormatFile(std::string file) {
  std::stringstream ss;
  ss << "\n File \"" << file << "\", ";
  return ss.str();
}

std::string FormatLine(const int64_t& line) {
  std::stringstream ss;
  ss << "line " << line << ",";
  return ss.str();
}

std::string FormatFunction(std::string function) {
  std::stringstream ss;
  ss << " in " << function;
  return ss.str();
}

std::string FormatErrorMsg(std::string error_msg, bool has_error_hint) {
  EraseErrorMsg(error_msg);
  if (!has_error_hint) { ShortenErrorMsg(error_msg); }
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

void FormatErrorStr(const std::shared_ptr<cfg::ErrorProto>& error) {
  std::string error_global = "";
  std::stringstream ss;
  for (auto stack_frame = error->mutable_stack_frame()->rbegin();
       stack_frame < error->mutable_stack_frame()->rend(); stack_frame++) {
    std::string error_file = FormatFile(*stack_frame->mutable_file());
    std::string error_line = FormatLine(*stack_frame->mutable_line());
    std::string error_function = FormatFunction(*stack_frame->mutable_function());
    std::string error_msg = FormatErrorMsg(*stack_frame->mutable_error_msg(),
                                           stack_frame == error->mutable_stack_frame()->rend() - 1);
    ss << error_file << error_line << error_function << error_msg;
  }
  std::string error_summary_and_msg = FormatErrorSummaryAndMsg(error);

  ss << error_global << "\n" << error_summary_and_msg;
  *MutErrorStr() = ss.str();
}

}  // namespace oneflow