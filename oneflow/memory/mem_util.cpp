#include "memory/mem_util.h"
#include <glog/logging.h>
namespace caffe {
// Return floor(log2(n)) for positive integer n.  Returns -1 if n == 0.
int32_t Log2Floor(uint32_t n) {
  if (n == 0) return -1;
  int32_t log = 0;
  uint32_t value = n;
  for (int32_t i = 4; i >= 0; --i) {
    int32_t shift = (1 << i);
    uint32_t x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  CHECK(value == 1);
  return log;
}
// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
// Log2Floor64() is defined in terms of Log2Floor32()
int32_t Log2Floor64(uint64_t n) {
  const uint32_t topbits = static_cast<uint32_t>(n >> 32);
  if (topbits == 0) {
    // Top bits are zero, so scan in bottom bits
    return Log2Floor(static_cast<uint32_t>(n));
  }
  else {
    return 32 + Log2Floor(topbits);
  }

}
int32_t Log2Ceiling(uint32_t n) {
  int32_t floor = Log2Floor(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}
int32_t Log2Ceiling64(uint64_t n) {
  int32_t floor = Log2Floor64(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}
size_t AlignSize(size_t bytes, size_t alignment) {
  CHECK(alignment % 256 == 0);
  if (bytes <= alignment) {
    return alignment;
  }
  size_t max_waste = 1uLL << (Log2Ceiling64(alignment));
  return (bytes + max_waste) & (~(max_waste - 1));
}
}  // namespace caffe