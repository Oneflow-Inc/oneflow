#include "oneflow/core/framework/multi_thread.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"

namespace oneflow {

namespace user_op {

void MultiThreadLoopInOpKernel(size_t num, std::function<void(size_t i)> Callback) {
  size_t thread_num = Global<ThreadMgr>::Get()->compute_thread_pool()->thread_num();
  thread_num = std::min(num, thread_num);
  BalancedSplitter bs(num, thread_num);
  BlockingCounter bc(thread_num);
  FOR_RANGE(size_t, range_id, 0, thread_num) {
    Global<ThreadMgr>::Get()->compute_thread_pool()->AddWork([&bc, &bs, range_id, Callback] {
      FOR_RANGE(size_t, i, bs.At(range_id).begin(), bs.At(range_id).end()) { Callback(i); }
      bc.Decrease();
    });
  }
  bc.WaitUntilCntEqualZero();
}

}  // namespace user_op

}  // namespace oneflow
