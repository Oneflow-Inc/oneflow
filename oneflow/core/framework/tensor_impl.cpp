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
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {
namespace one {

namespace {

Maybe<const ParallelDesc> MakeParallelDescByDevice(const Device& device) {
  int64_t machine_id = GlobalProcessCtx::Rank();
  int64_t device_id = device.device_id();
  std::string machine_device_id = std::to_string(machine_id) + ":" + std::to_string(device_id);
  std::vector<std::string> machine_device_ids({machine_device_id});
  Maybe<cfg::ParallelConf> conf = MakeParallelConf(device.of_type(), machine_device_ids);
  return std::make_shared<const ParallelDesc>(JUST(conf));
}

}  // namespace

Maybe<void> MirroredTensorImpl::set_device(const std::shared_ptr<const Device>& device) {
  device_ = device;
  parallel_desc_ = JUST(MakeParallelDescByDevice(*device));
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
