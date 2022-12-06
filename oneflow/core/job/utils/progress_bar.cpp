/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/utils/progress_bar.h"
#include "oneflow/core/job/graph_scope_vars.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

Maybe<void> LogProgress(const std::string& task_name, bool is_end) {
  const bool log_progress =
      GetGraphDebugMode() || ThreadLocalEnvBool<ONEFLOW_NNGRAPH_ENABLE_PROGRESS_BAR>();
  if (!log_progress || OF_PREDICT_FALSE(GlobalProcessCtx::Rank() != 0)) {
    return Maybe<void>::Ok();
  }

  const static thread_local uint64_t progress_total_num = 60;
  static thread_local uint64_t progress_cnt = 1;
  static constexpr char clear_line[] =
      "                                                                         \r";

  auto const& limited_str = task_name.size() > 60 ? task_name.substr(0, 60) : task_name;
  std::cout << clear_line << "[" << progress_cnt << "/" << progress_total_num << "]" << limited_str
            << "\r" << std::flush;
  if (is_end) {
    progress_cnt = 0;
    std::cout << clear_line << std::endl << std::flush;
  }
  ++progress_cnt;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
