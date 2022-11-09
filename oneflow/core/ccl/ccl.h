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
#ifndef ONEFLOW_CORE_CCL_CCL_H_
#define ONEFLOW_CORE_CCL_CCL_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

class ParallelDesc;
class TransportToken;

// collective communication library
namespace ccl {

Maybe<void> CpuSend(const void* in, size_t buffer_size, int64_t dst);

Maybe<void> CpuRecv(void* out, size_t buffer_size, int64_t src);

Maybe<void> CpuBroadcast(const void* in, void* out, size_t buffer_size, int64_t root,
                         Symbol<ParallelDesc> parallel_desc, const TransportToken& transport_token);

}  // namespace ccl

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CCL_CCL_H_
