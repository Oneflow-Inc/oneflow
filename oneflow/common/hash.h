#ifndef _COMMON_HASH_H_
#define _COMMON_HASH_H_

#include <stdint.h>

namespace caffe {
namespace hash {
// NOTE(Chonglin): Currently use Elf Hash because it is simple
// And we don't have two many keys to hash, so it's not easy to
// cause collision.
uint32_t ElfHash(const unsigned char *s);
}  // namespace hash
}  // namespace caffe

#endif
