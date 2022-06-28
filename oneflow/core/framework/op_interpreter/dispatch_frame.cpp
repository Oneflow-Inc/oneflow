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
#include "oneflow/core/framework/op_interpreter/dispatch_frame.h"
#include <string>

namespace oneflow {

/* static */ std::string* DispatchFrame::get_str_ptr() {
  static thread_local std::string frame_str = "";
  return &frame_str;
}

/* static */ const std::string& DispatchFrame::get_str() { return *get_str_ptr(); }

/* static */ void DispatchFrame::set_str(const std::string& str) { *get_str_ptr() = str; }

}  // namespace oneflow
