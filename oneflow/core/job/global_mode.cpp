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
#include "oneflow/core/framework/device.h"

namespace oneflow {

Symbol<ParallelDesc> GetGlobalParallelDescFromDevice(const Optional<Symbol<Device>>& device) {
  auto parallel_desc = GlobalMode::parallel_desc();
  if (device.has_value()) {
    const auto& device_type = device.value_or(Symbol<Device>())->type();
    if (parallel_desc->parallel_conf().device_tag() != device_type) {
      ParallelConf parallel_conf = parallel_desc->parallel_conf();
      parallel_conf.set_device_tag(device_type);
      parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
    }
  }
  return parallel_desc;
}

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
