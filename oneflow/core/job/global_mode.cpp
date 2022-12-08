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

#include "oneflow/core/job/global_mode.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

/* static */ bool* GlobalMode::get_mode_ptr() {
  thread_local bool mode = false;
  return &mode;
}
/* static */ bool GlobalMode::is_enabled() { return *get_mode_ptr(); }
/* static */ void GlobalMode::set_enabled(bool enabled) { *get_mode_ptr() = enabled; }

/* static */ Symbol<NdSbp>* GlobalMode::get_nd_sbp_ptr() {
  thread_local Symbol<NdSbp> nd_sbp;
  return &nd_sbp;
}
/* static */ Symbol<NdSbp> GlobalMode::nd_sbp() { return *get_nd_sbp_ptr(); }
/* static */ void GlobalMode::set_nd_sbp(Symbol<NdSbp> nd_sbp) { *get_nd_sbp_ptr() = nd_sbp; }

/* static */ Symbol<ParallelDesc>* GlobalMode::get_parallel_desc_ptr() {
  thread_local Symbol<ParallelDesc> parallel_desc;
  return &parallel_desc;
}
/* static */ Symbol<ParallelDesc> GlobalMode::parallel_desc() { return *get_parallel_desc_ptr(); }
/* static */ void GlobalMode::set_parallel_desc(Symbol<ParallelDesc> parallel_desc) {
  *get_parallel_desc_ptr() = parallel_desc;
}

}  // namespace oneflow
