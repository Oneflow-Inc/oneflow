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

#ifndef ONEFLOW_CORE_JOB_GLOBAL_MODE_H_
#define ONEFLOW_CORE_JOB_GLOBAL_MODE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.pb.h"

namespace oneflow {

Symbol<ParallelDesc> GetGlobalParallelDescFromDevice(const Optional<Symbol<Device>>& device);
class GlobalMode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GlobalMode);
  GlobalMode() = default;
  ~GlobalMode() = default;

  static bool is_enabled();
  static Symbol<NdSbp> nd_sbp();
  static Symbol<ParallelDesc> parallel_desc();

  class Guard {
   public:
    explicit Guard(bool enabled)
        : prev_mode_(GlobalMode::is_enabled()),
          prev_nd_sbp_(GlobalMode::nd_sbp()),
          prev_parallel_desc_(GlobalMode::parallel_desc()) {
      CHECK(!enabled);
      GlobalMode::set_enabled(enabled);
    }
    explicit Guard(bool enabled, Symbol<NdSbp> nd_sbp, Symbol<ParallelDesc> parallel_desc)
        : prev_mode_(GlobalMode::is_enabled()),
          prev_nd_sbp_(GlobalMode::nd_sbp()),
          prev_parallel_desc_(GlobalMode::parallel_desc()) {
      GlobalMode::set_enabled(enabled);
      if (enabled) {
        GlobalMode::set_nd_sbp(nd_sbp);
        GlobalMode::set_parallel_desc(parallel_desc);
      }
    }
    ~Guard() {
      GlobalMode::set_enabled(prev_mode_);
      GlobalMode::set_nd_sbp(prev_nd_sbp_);
      GlobalMode::set_parallel_desc(prev_parallel_desc_);
    }

   private:
    bool prev_mode_;
    Symbol<NdSbp> prev_nd_sbp_;
    Symbol<ParallelDesc> prev_parallel_desc_;
  };

 private:
  static bool* get_mode_ptr();
  static Symbol<NdSbp>* get_nd_sbp_ptr();
  static Symbol<ParallelDesc>* get_parallel_desc_ptr();

  static void set_enabled(bool enabled);
  static void set_nd_sbp(Symbol<NdSbp> nd_sbp);
  static void set_parallel_desc(Symbol<ParallelDesc> parallel_desc);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_GLOBAL_MODE_H_
