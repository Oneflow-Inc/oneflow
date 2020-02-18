#include "oneflow/core/framework/multi_thread.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

namespace user_op {

void MultiThreadLoopInOpKernel(size_t num, std::function<void(size_t i)> Callback) {
  MultiThreadLoop(num, Callback);
}

}  // namespace user_op

}  // namespace oneflow
