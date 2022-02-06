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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/placement_utils.h"
#include "oneflow/core/framework/parallel_conf_util.h"

namespace oneflow {

Maybe<Symbol<ParallelDesc>> ReplacePlacementDeviceTag(Symbol<ParallelDesc> parallel_desc,
                                                      const std::string& device_type) {
  static const HashMap<std::string, std::string> type2device_tag{{"cpu", "cpu"}, {"cuda", "gpu"}};
  ParallelConf parallel_conf(parallel_desc->parallel_conf());
  parallel_conf.set_device_tag(JUST(MapAt(type2device_tag, device_type)));
  return SymbolOf(ParallelDesc(parallel_conf));
}

Maybe<void> TouchConsistentTensor(const std::shared_ptr<one::Tensor>& tensor) {
  CHECK_OR_RETURN(tensor->is_consistent());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
