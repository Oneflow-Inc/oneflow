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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_DISPATCH_FRAME_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_DISPATCH_FRAME_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class DispatchFrame {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DispatchFrame);
  DispatchFrame() = delete;
  ~DispatchFrame() = delete;

  static const std::string& get_str();
  static void set_str(const std::string& str);

  class Guard {
   public:
    explicit Guard(const std::string& frame_str) : prev_frame_str_(DispatchFrame::get_str()) {
      DispatchFrame::set_str(frame_str);
    }
    ~Guard() { DispatchFrame::set_str(prev_frame_str_); }

   private:
    std::string prev_frame_str_;
  };

 private:
  static std::string* get_str_ptr();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_DISPATCH_FRAME_H_
