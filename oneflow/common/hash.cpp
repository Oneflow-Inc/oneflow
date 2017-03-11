#include "common/hash.h"

namespace caffe {
namespace hash {
uint32_t ElfHash(const unsigned char *s) {
  uint32_t h = 0, high;
  while (*s) {
    h = (h << 4) + *s++;
    if (high = h & 0xF0000000)
      h ^= high >> 24;
    h &= ~high;
  }
  return h;
}
}  // namespace hash
}  // namespace caffe
