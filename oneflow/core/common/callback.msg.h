#ifndef ONEFLOW_CORE_COMMON_CALLBACK_MSG_H_
#define ONEFLOW_CORE_COMMON_CALLBACK_MSG_H_

#include <functional>
#include "oneflow/core/common/object_msg.h"

namespace oneflow {

// clang-format off
OBJECT_MSG_BEGIN(CallbackMsg);
  // methods
  PUBLIC void __Init__() {}
  PUBLIC void __Init__(const std::function<void()>& callback) { *mut_callback() = callback; }

  // fields
  OBJECT_MSG_DEFINE_STRUCT(std::function<void()>, callback);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(callback_link);
OBJECT_MSG_END(CallbackMsg);
// clang-format on

using CallbackMsgListPtr = OBJECT_MSG_LIST_PTR(CallbackMsg, callback_link);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CALLBACK_MSG_H_
