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
#ifndef ONEFLOW_CORE_EAGER_TRANSPORT_PARALLEL_CONF_H_
#define ONEFLOW_CORE_EAGER_TRANSPORT_PARALLEL_CONF_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/eager/parallel_conf_group.pb.h"

namespace oneflow {
namespace eager {

// Devices are part of src_parallel_desc for all ParallelConf objects in ParallelConfGroupList
// The src_parallel_desc must be reconstructed by ParallelConfGroupList in a way similar to
// numpy.vstack There is only one device in source machine and destination machine for all
// ParallelConf objects in ParallelConfGroupList source machines and destination machines are shared
// among ParallelConf objects under the same ParallelConfGroup e.g.
//    src: {"0:0-3"}
//    dst: {"0:2-3", "1:0-1"}
//    ret: [[{"0:0"}, {"0:1"}], [{"0:2"}, "0:3"]]
Maybe<ParallelConfGroupList> MakeTransportInstructionParallelConfs(
    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc);

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_TRANSPORT_PARALLEL_CONF_H_
