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

#include "oneflow/core/common/error_util.h"

namespace oneflow {

std::string& GetErrorStr() {
  thread_local static std::string error_str = "";
  return error_str;
}

namespace {

void SpaceStrip(std::string& str) {
  str.erase(0, str.find_first_not_of(" "));
  str.erase(str.find_last_not_of(" ") + 1);
}

bool IsLetterNumberOrUnderline(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
}

void ErrorMsgErase(std::string& str) {
  if (str.size() == 0) { return; }
  SpaceStrip(str);
  // strip bracket
  if (str.at(0) != '(') { return; }
  str = str.substr(1, str.size() - 2);
}

void ErrorMsgShorten(std::string& str) {
  const int num_displayed_char = 150;
  if (str.size() == 0) { return; }
  SpaceStrip(str);
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
  str = str.substr(0, first_index) + " ... " + str.substr(last_index);
}

std::string LocationFormat(std::string location) {
  int index = location.find(":");
  location.erase(index, 1);
  location.insert(index, "\", line ");
  return "\n  File \"" + location + ",";
}

std::string FunctionFormat(std::string function) { return " in " + function; }

std::string ErrorMsgFormat(std::string error_msg, bool has_error_hint) {
  ErrorMsgErase(error_msg);
  if (!has_error_hint) { ErrorMsgShorten(error_msg); }
  return "\n    " + error_msg;
}

std::string ErrorSummaryAndMsgFormat(const std::shared_ptr<cfg::ErrorProto>& error) {
  std::string error_summary_and_msg = "";
  if (error->has_error_summary()) { error_summary_and_msg += error->error_summary(); }
  if (error->has_msg()) {
    error_summary_and_msg +=
        (error_summary_and_msg.size() != 0 ? ", " + error->msg() : error->msg());
  }
  return error_summary_and_msg;
}

}  // namespace

void ErrorStrFormat(const std::shared_ptr<cfg::ErrorProto>& error) {
  std::string error_global = "";
  for (auto stack_frame = error->mutable_stack_frame()->rbegin();
       stack_frame < error->mutable_stack_frame()->rend(); stack_frame++) {
    std::string error_file = LocationFormat(*stack_frame->mutable_location());
    std::string error_function = FunctionFormat(*stack_frame->mutable_function());
    std::string error_msg = ErrorMsgFormat(*stack_frame->mutable_error_msg(),
                                           stack_frame == error->mutable_stack_frame()->rend() - 1);
    error_global += (error_file + error_function + error_msg);
  }
  std::string error_summary_and_msg = ErrorSummaryAndMsgFormat(error);

  GetErrorStr() += (error_global + "\n" + error_summary_and_msg);
}

}  // namespace oneflow