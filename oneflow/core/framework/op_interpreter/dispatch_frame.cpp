#include "oneflow/core/framework/op_interpreter/dispatch_frame.h"
#include <string>

namespace oneflow {

/* static */ std::string* DispatchFrame::get_str_ptr() {
  static thread_local std::string frame_str = "";
  return &frame_str;
}

/* static */ std::string DispatchFrame::get_str() { return *get_str_ptr(); }

/* static */ void DispatchFrame::set_str(const std::string& str) { *get_str_ptr() = str; }

}  // namespace oneflow
