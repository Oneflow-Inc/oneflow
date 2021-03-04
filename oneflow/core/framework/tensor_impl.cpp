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
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {
namespace one {

namespace {

Maybe<const ParallelDesc> MakeParallelDescByDevice(const Device& device) {
  int64_t machine_id = GlobalProcessCtx::Rank();
  int64_t device_id = device.device_id();
  std::string machine_device_id = std::to_string(machine_id) + ":" + std::to_string(device_id);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(device.of_type());
  parallel_conf.add_device_name(machine_device_id);
  return std::make_shared<const ParallelDesc>(parallel_conf);
}

}  // namespace

Maybe<void> MirroredTensorImpl::set_device(const std::shared_ptr<const Device>& device) {
  device_ = device;
  parallel_desc_ = JUST(MakeParallelDescByDevice(*device));
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
