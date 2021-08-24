#include "oneflow/core/common/thread_local_callback.h"

namespace oneflow {

namespace blocking {

using StackInfoCallbackType = std::function<std::string()>;

thread_local StackInfoCallbackType StackInfoCallback;

void RegisterStackInfoCallback(const StackInfoCallbackType& Callback) {
  StackInfoCallback = Callback;
}
StackInfoCallbackType GetStackInfoCallback() { return StackInfoCallback; }
void ClearStackInfoCallback() {}

}  // namespace blocking

}  // namespace oneflow
