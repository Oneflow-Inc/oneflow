#include <cstdlib>
#include "oneflow/core/vm/cpu_allocator.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void CpuAllocator::Allocate(char** mem_ptr, std::size_t size) {
  *mem_ptr = reinterpret_cast<char*>(std::malloc(size));
}

void CpuAllocator::Deallocate(char* mem_ptr, std::size_t size) { std::free(mem_ptr); }

COMMAND(Global<CpuAllocator>::SetAllocated(new CpuAllocator()));

}  // namespace vm
}  // namespace oneflow
