#include <cstring>
#include "oneflow/core/kernel/rle_util.h"

extern "C" {

#include "maskApi.h"
}

namespace oneflow {

size_t RleEncode(uint32_t* buf, const uint8_t* mask, size_t h, size_t w) {
  RLE rle;
  rleEncode(&rle, mask, h, w, 1);
  size_t len = rle.m;
  std::memcpy(buf, rle.cnts, len * sizeof(buf[0]));
  rleFree(&rle);
  return len;
}

}  // namespace oneflow
