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
#include <cstdlib>
#include "oneflow/core/vm/cpu_allocator.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void CpuAllocator::Allocate(char** mem_ptr, std::size_t size) {
  *mem_ptr = reinterpret_cast<char*>(aligned_alloc(kHostAlignSize, size));
}

void CpuAllocator::Deallocate(char* mem_ptr, std::size_t size) { std::free(mem_ptr); }

COMMAND(Singleton<CpuAllocator>::SetAllocated(new CpuAllocator()));

}  // namespace vm
}  // namespace oneflow
