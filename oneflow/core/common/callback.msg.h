#include <functional>
#include "oneflow/core/common/object_msg.h"

namespace oneflow {

// clang-format off
OBJECT_MSG_BEGIN(CallbackMsg);
  // fields
  OBJECT_MSG_DEFINE_STRUCT(std::function<void()>, callback);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(callback_link);
OBJECT_MSG_END(CallbackMsg);
// clang-format on

}  // namespace oneflow
