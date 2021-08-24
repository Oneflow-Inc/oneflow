#include "oneflow/core/common/thread_local_callback.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace blocking {

using StackInfoCallbackType = std::function<std::string()>;

thread_local StackInfoCallbackType StackInfoCallback;

void RegisterStackInfoCallback(const StackInfoCallbackType& Callback) {
  StackInfoCallback = Callback;
}
StackInfoCallbackType GetStackInfoCallback() { return StackInfoCallback; }
std::string GetStackInfo() {
  return "[rank=" + std::to_string(GlobalProcessCtx::Rank()) + "]"
         + " blocking detected. Python stack:\n" + GetStackInfoCallback()();
}
void ClearStackInfoCallback() {}

}  // namespace blocking

}  // namespace oneflow
