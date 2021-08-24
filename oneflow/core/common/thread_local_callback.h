#ifndef ONEFLOW_CORE_COMMON_THREAD_LOCAL_CALLBACK_H_
#define ONEFLOW_CORE_COMMON_THREAD_LOCAL_CALLBACK_H_

#include <functional>

namespace oneflow {

namespace blocking {

using StackInfoCallbackType = std::function<std::string()>;

void RegisterStackInfoCallback(const StackInfoCallbackType& Callback);
StackInfoCallbackType GetStackInfoCallback();
std::string GetStackInfo();
void ClearStackInfoCallback();

}  // namespace blocking

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_THREAD_LOCAL_CALLBACK_H_
