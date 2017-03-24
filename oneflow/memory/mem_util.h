#ifndef _MEM_UTIL_H_
#define _MEM_UTIL_H_
#include <cstdint>
namespace oneflow {
int32_t Log2Floor(uint32_t n);
int32_t Log2Floor(uint64_t n);
int32_t Log2Ceiling(uint32_t n);
int32_t Log2Ceiling64(uint64_t n);
// Get the number of bytes after being aligned with the value |alignment|.
// The default value of |alignment| is 256: 
// https://devblogs.nvidia.com/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/
// We require the value |alignment| is multiples of 256.
// Borrow the idea and implementation from TensorFlow
size_t AlignSize(size_t bytes, size_t alignment = 256);
}  // namespace oneflow
#endif  // _MEM_UTIL_H_
