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
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_

#include "oneflow/core/common/blas.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

class Blob;
class MemoryCase;
class StreamContext;

void AutoMemcpy(ep::Stream* stream, void* dst, const void* src, size_t sz,
                const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case);
void AutoMemcpy(ep::Stream* stream, Blob* dst, const Blob* src);
void SyncAutoMemcpy(ep::Stream* stream, void* dst, const void* src, size_t sz,
                    const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case);
void AutoMemset(ep::Stream* stream, void* dst, const char value, size_t sz,
                const MemoryCase& dst_mem_case);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
