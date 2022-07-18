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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/memset.h"

namespace oneflow {

void AutoMemcpy(ep::Stream* stream, void* dst, const void* src, size_t sz,
                const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case) {
  ep::primitive::MemcpyKind kind{};
  if (stream->device_type() == DeviceType::kCPU) {
    CHECK(memory::IsHostMem(src_mem_case));
    CHECK(memory::IsHostMem(dst_mem_case));
    kind = ep::primitive::MemcpyKind::kDtoD;
  } else {
    if (memory::IsHostMem(src_mem_case)) {
      CHECK(!memory::IsHostMem(dst_mem_case));
      kind = ep::primitive::MemcpyKind::kHtoD;
    } else if (memory::IsHostMem(dst_mem_case)) {
      CHECK(!memory::IsHostMem(src_mem_case));
      kind = ep::primitive::MemcpyKind::kDtoH;
    } else {
      kind = ep::primitive::MemcpyKind::kDtoD;
    }
  }
  std::unique_ptr<ep::primitive::Memcpy> primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(stream->device_type(), kind);
  CHECK(primitive);
  primitive->Launch(stream, dst, src, sz);
}

void AutoMemcpy(ep::Stream* stream, Blob* dst, const Blob* src) {
  const size_t body_bytes = src->ByteSizeOfBlobBody();
  CHECK_EQ(dst->ByteSizeOfBlobBody(), body_bytes);
  AutoMemcpy(stream, dst->mut_dptr(), src->dptr(), body_bytes, dst->mem_case(), src->mem_case());
}

void SyncAutoMemcpy(ep::Stream* stream, void* dst, const void* src, size_t sz,
                    const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case) {
  AutoMemcpy(stream, dst, src, sz, dst_mem_case, src_mem_case);
  CHECK_JUST(stream->Sync());
}

void AutoMemset(ep::Stream* stream, void* dst, const char value, size_t sz,
                const MemoryCase& /*dst_mem_case*/) {
  std::unique_ptr<ep::primitive::Memset> primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(stream->device_type());
  primitive->Launch(stream, dst, value, sz);
}

}  //  namespace oneflow
