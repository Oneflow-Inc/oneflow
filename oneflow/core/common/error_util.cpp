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

#include <stack>
#include "oneflow/core/common/error_util.h"

namespace oneflow {

std::string& ErrorStrGet() {
  thread_local static std::string error_str = "";
  return error_str;
}

namespace {

void SpaceStrip(std::string& str) {
  str.erase(0, str.find_first_not_of(" "));
  str.erase(str.find_last_not_of(" ") + 1);
}

void ErrorMsgEraseMaybe(std::string& str) {
  if (str.size() == 0) { return; }
  SpaceStrip(str);
  // maybe strip bracket
  if (str.at(0) != '(') { return; }
  std::stack<char> left_bracket;
  left_bracket.push(str.at(0));
  for (auto c = str.begin() + 1; c < str.end(); c++) {
    if (*c == '(') { left_bracket.push(*c); }
    if (*c == ')') {
      left_bracket.pop();
      if (left_bracket.empty()) {
        str.erase(c);
        str = str.substr(1);
        break;
      }
    }
  }
}

void ErrorMsgShortenMaybe(std::string& str) {
  std::unordered_map<int, int> delim_index2length;
  SpaceStrip(str);
  int word_num = 0;
  for (int i = 0; i < str.size(); i++) {
    if (str.at(i) == ' ' || str.at(i) == '.') {
      delim_index2length.insert(std::make_pair(word_num, i));
      word_num++;
    } else if (str.at(i) == '-' && str.at(i + 1) == '>') {
      delim_index2length.insert(std::make_pair(word_num, i));
      word_num++;
    }
  }
  if (word_num > 10) {
    str = str.substr(0, delim_index2length.at(4)) + " ... "
          + str.substr(delim_index2length.at(word_num - 4) + 1);
  }
}

std::string LocationFormat(std::string location) {
  std::size_t index = location.find("line");
  return "\n  File \"" + location.substr(0, index) + "\", " + location.substr(index, 4) + " "
         + location.substr(index + 4) + ",";
}

std::string FunctionFormat(std::string function) { return " in " + function; }

std::string ErrorMsgFormat(std::string error_msg) {
  ErrorMsgEraseMaybe(error_msg);
  ErrorMsgShortenMaybe(error_msg);
  return "\n    " + error_msg;
}

}  // namespace

void ErrorStrFormat(const std::shared_ptr<cfg::ErrorProto>& error) {
  for (auto stack_frame = error->mutable_stack_frame()->rbegin();
       stack_frame < error->mutable_stack_frame()->rend(); stack_frame++) {
    std::string error_file = LocationFormat(*stack_frame->mutable_location());
    std::string error_function = FunctionFormat(*stack_frame->mutable_function());
    std::string error_msg = ErrorMsgFormat(*stack_frame->mutable_error_msg());
    ErrorStrGet() += (error_file + error_function + error_msg);
  }
}

}  // namespace oneflow