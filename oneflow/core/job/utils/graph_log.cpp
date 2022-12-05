
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/utils/graph_log.h"

namespace oneflow {
Maybe<void> LogProgress() {
  const static thread_local uint64_t progress_total_num = 66;
  static thread_local uint64_t progress_cnt = 1;
  if (OF_PREDICT_FALSE(progress_cnt==1)) {
    std::cout << "nn.Graph compilation has " << progress_total_num << " tasks, progress: ";
  }
  std::cout << progress_cnt << "|";
  ++progress_cnt;
  return Maybe<void>::Ok();
}

}  // namespace oneflow