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
#include "oneflow/core/common/util.h"

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

Maybe<std::string> ShortenMsg(std::string str) {
  // 150 characters is the threshold
  const int character_num_threshold = 150;
  const int num_displayed_char = 50;
  if (str.size() == 0) { return str; }
  // strip space when JUST(  xx  );
  str = StripSpace(str);
  if (str.size() < character_num_threshold) { return str; }

  // left part whose number of characters is just over 50
  int left_index = num_displayed_char;
  if (IsLetterNumberOrUnderline(str.at(left_index))) {
    for (; left_index < str.size(); left_index++) {
      if (!IsLetterNumberOrUnderline(str.at(left_index))) { break; }
    }
  } else {
    for (; left_index < str.size(); left_index++) {
      if (IsLetterNumberOrUnderline(str.at(left_index))) { break; }
    }
  }

  // right part whose number of characters is just over 50
  int right_index = str.size() - num_displayed_char;
  if (IsLetterNumberOrUnderline(str.at(right_index))) {
    for (; right_index >= 0; right_index--) {
      if (!IsLetterNumberOrUnderline(str.at(right_index))) {
        right_index++;
        break;
      }
    }
  } else {
    for (; right_index >= 0; right_index--) {
      if (IsLetterNumberOrUnderline(str.at(right_index))) {
        right_index++;
        break;
      }
    }
  }
  // a long word of more than 150
  if (right_index - left_index < 50) { return str; }
  std::stringstream ss;
  CHECK_OR_RETURN(left_index >= 0);
  CHECK_OR_RETURN(left_index < str.size());
  ss << str.substr(0, left_index);
  ss << " ... ";
  CHECK_OR_RETURN(right_index >= 0);
  CHECK_OR_RETURN(right_index < str.size());
  ss << str.substr(right_index);
  return ss.str();
}

// file info in stack frame
std::string FormatFileOfStackFrame(const std::string& file) {
  std::stringstream ss;
  ss << "\n  File \"" << file << "\", ";
  return ss.str();
}

// line info in stack frame
std::string FormatLineOfStackFrame(const int64_t& line) {
  std::stringstream ss;
  ss << "line " << line << ",";
  return ss.str();
}

// function info in stack frame
std::string FormatFunctionOfStackFrame(const std::string& function) {
  std::stringstream ss;
  ss << " in " << function;
  return ss.str();
}

// msg in stack frame
Maybe<std::string> FormatMsgOfStackFrame(std::string error_msg, bool is_last_stack_frame) {
  error_msg = StripBrackets(error_msg);
  if (!is_last_stack_frame) { error_msg = *JUST(ShortenMsg(error_msg)); }
  // error_msg of last stack frame come from "<<"
  if (is_last_stack_frame) { error_msg = StripSpace(error_msg); }
  std::stringstream ss;
  ss << "\n    " << error_msg;
  return ss.str();
}

// the error_summary and msg in error proto
std::string FormatErrorSummaryAndMsgOfErrorProto(const std::shared_ptr<cfg::ErrorProto>& error) {
  std::stringstream ss;
  if (error->has_error_summary()) { ss << error->error_summary(); }
  if (error->has_msg()) { ss << (ss.str().size() != 0 ? "\n" + error->msg() : error->msg()); }
  return ss.str();
}

// the msg in error type instance.
Maybe<std::string> FormatMsgOfErrorType(const std::shared_ptr<cfg::ErrorProto>& error) {
  std::stringstream ss;
  CHECK_NE(error->error_type_case(), cfg::ErrorProto::ERROR_TYPE_NOT_SET);
  switch (error->error_type_case()) {
    case cfg::ErrorProto::kInputDeviceNotMatchError:
      ss << error->input_device_not_match_error().DebugString();
      break;
    case cfg::ErrorProto::kMemoryZoneOutOfMemoryError:
      ss << error->memory_zone_out_of_memory_error().DebugString();
      break;
    case cfg::ErrorProto::kMultipleOpKernelsMatchedError:
      ss << error->multiple_op_kernels_matched_error().DebugString();
      break;
    case cfg::ErrorProto::kOpKernelNotFoundError:
      ss << error->op_kernel_not_found_error().DebugString();
      break;
    case cfg::ErrorProto::kConfigResourceUnavailableError:
      ss << error->config_resource_unavailable_error().DebugString();
      break;
    case cfg::ErrorProto::kConfigAssertFailedError:
      ss << error->config_assert_failed_error().DebugString();
      break;
    default: ss << "";
  }
  return ss.str();
}

}  // namespace

Maybe<std::string> FormatErrorStr(const std::shared_ptr<cfg::ErrorProto>& error) {
  std::stringstream ss;
  // Get msg from stack frame of error proto
  for (auto stack_frame = error->mutable_stack_frame()->rbegin();
       stack_frame < error->mutable_stack_frame()->rend(); stack_frame++) {
    ss << FormatFileOfStackFrame(*stack_frame->mutable_file())
       << FormatLineOfStackFrame(*stack_frame->mutable_line())
       << FormatFunctionOfStackFrame(*stack_frame->mutable_function())
       << *JUST(FormatMsgOfStackFrame(*stack_frame->mutable_error_msg(),
                                      stack_frame == error->mutable_stack_frame()->rend() - 1));
  }
  // Get msg from error summary and msg of error proto
  std::string error_summary_and_msg_of_error_proto = FormatErrorSummaryAndMsgOfErrorProto(error);
  if (error_summary_and_msg_of_error_proto.size() != 0) {
    ss << "\n" << error_summary_and_msg_of_error_proto;
  }
  // Get msg from error type of error proto
  std::string msg_of_error_type = *JUST(FormatMsgOfErrorType(error));
  if (msg_of_error_type.size() != 0) { ss << "\n" << msg_of_error_type; }

  return ss.str();
}

}  // namespace oneflow
