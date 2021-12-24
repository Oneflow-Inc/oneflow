#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_DISPATCH_FRAME_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_DISPATCH_FRAME_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class DispatchFrame{
 public:
  OF_DISALLOW_COPY_AND_MOVE(DispatchFrame);
  DispatchFrame() = delete;
  ~DispatchFrame() = delete;

  static std::string get_str();

  class Guard {
   public:
    explicit Guard(const std::string& frame_str) : prev_frame_str_(DispatchFrame::get_str()) { DispatchFrame::set_str(frame_str); }
    ~Guard() { DispatchFrame::set_str(prev_frame_str_); }

   private:
    std::string prev_frame_str_;
  };

 private:
  static std::string* get_str_ptr();
  static void set_str(const std::string& str);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_DISPATCH_GUARD_H_